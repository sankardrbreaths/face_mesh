from typing import Tuple, Union
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import time
import pandas as pd
import os


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(image, detection_result):
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    POINT_RADIUS = 2  # Font size
    POINT_FONT_COLOR = (0, 255, 0)  # Green color in BGR
    POINT_FONT_THICKNESS = 2 # Line thickness
    
    NO_FONT_SCALE = 1  # Font size
    NO_FONT_COLOR = (0, 0, 255)  # Red color in BGR
    NO_FONT_THICKNESS = 2 # Line thickness

    # Draw keypoints
    # key_point_name = ['right_eye_x', 'right_eye_y', 'left_eye_x', 'left_eye_y', 'nose_x', 'nose_y', 'mouth_x', 'mouth_y', 'right_ear_x', 'right_ear_y', 'left_ear_x', 'left_ear_y']

    key_point_name = ['right_eye', 'left_eye', 'nose', 'mouth', 'right_ear', 'left_ear']

    keypoint_px  = [_normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height) for keypoint in detection.keypoints]
    
    keypoint_dict = dict(zip(key_point_name, keypoint_px))

    for name, position in keypoint_dict.items():
      cv2.circle(annotated_image, position, POINT_FONT_THICKNESS, POINT_FONT_COLOR, POINT_RADIUS)

      # Plot the number at the keypoint
      cv2.putText(annotated_image, name, position, FONT, NO_FONT_SCALE, NO_FONT_COLOR, NO_FONT_THICKNESS, cv2.LINE_AA)


    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image, keypoint_dict

def calculate_distance(point1, point2):
  return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def face_orient_classifier(keypoint_dict):
  angle_rad_left_eye = math.degrees(math.atan2(keypoint_dict['nose'][1] - keypoint_dict['left_eye'][1], keypoint_dict['nose'][0] - keypoint_dict['left_eye'][0]))
  angle_rad_right_eye = math.degrees(math.atan2(keypoint_dict['nose'][1] - keypoint_dict['right_eye'][1], keypoint_dict['nose'][0] - keypoint_dict['right_eye'][0]))

  if angle_rad_left_eye > 90 > angle_rad_right_eye :
    return 'front'
  elif angle_rad_left_eye and angle_rad_right_eye > 90 or angle_rad_left_eye and angle_rad_right_eye < 90:
    return 'profile'
  else:
    return 'not_sure'
     

def main():

  base_options = python.BaseOptions(model_asset_path='detector.tflite')
  options = vision.FaceDetectorOptions(base_options=base_options)
  detector = vision.FaceDetector.create_from_options(options)

  directory = 'Data/Data_SVM/profile'
  pred_directory = 'Data/prediction_profile'
  os.makedirs(pred_directory, exist_ok=True)
  key_point_name = ['right_eye', 'left_eye', 'nose', 'mouth', 'right_ear', 'left_ear']


  data = {key: [] for key in key_point_name}
  for root, dirs, files in os.walk(directory):
      for file in files:
          if file.lower().endswith(('.jpg', '.png')):
              img_path = os.path.join(root, file)

              image = mp.Image.create_from_file(img_path)

              # STEP 4: Detect faces in the input image.
              face_detection_result = detector.detect(image)

              # STEP 5: Process the detection result. In this case, visualize it.
              image_copy = np.copy(image.numpy_view())
              annotated_image, keypoint_dict = visualize(image_copy, face_detection_result)

              # left_eye_dist_from_nose = calculate_distance(keypoint_dict['nose'], keypoint_dict['left_eye'])
              # right_eye_dist_from_nose = calculate_distance(keypoint_dict['nose'], keypoint_dict['right_eye'])

              pred = face_orient_classifier(keypoint_dict)
              print(pred)
              keypoint_dict['img']= img_path.split('/')[-1]
              keypoint_dict['class_gt']= img_path.split('/')[-2]
              keypoint_dict['class_pred']= pred
              img = Image.fromarray(annotated_image)
              img.save(os.path.join(pred_directory, file))

if __name__ == "__main__":
  main()