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
from PIL import Image, ExifTags
import io

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


def _visualize(image, detection_result):
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
    text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image


def _face_orient_classifier(keypoint_dict)->str:
  angle_left_eye = math.degrees(math.atan2(keypoint_dict['nose'][1] - keypoint_dict['left_eye'][1], keypoint_dict['nose'][0] - keypoint_dict['left_eye'][0]))
  angle_right_eye = math.degrees(math.atan2(keypoint_dict['nose'][1] - keypoint_dict['right_eye'][1], keypoint_dict['nose'][0] - keypoint_dict['right_eye'][0]))

  if angle_left_eye > 90 > angle_right_eye :
    prediction = 'front'
  elif angle_left_eye and angle_right_eye >= 90 or angle_left_eye and angle_right_eye <= 90:
    prediction = 'profile'

  return prediction


def _correct_rotation(img: Image)->Image:
  try:
      if hasattr(img, '_getexif'): # only present in PIL images
          for orientation in ExifTags.TAGS.keys():
              if ExifTags.TAGS[orientation] == 'Orientation':
                  break
          e = img._getexif()
          if e is not None:
              exif = dict(e.items())
              orientation = exif[orientation]

              if orientation == 3:
                  img = img.rotate(180, expand=True)
              elif orientation == 6:
                  img = img.rotate(270, expand=True)
              elif orientation == 8:
                  img = img.rotate(90, expand=True)
      return img
  except:
      return img
    

def detect_face(img: np.array)->str:
  base_options = python.BaseOptions(model_asset_path='detector.tflite')
  options = vision.FaceDetectorOptions(base_options=base_options)
  detector = vision.FaceDetector.create_from_options(options)
  image = mp.Image(data=np.asarray(img), image_format=mp.ImageFormat.SRGB)

  face_detection_result = detector.detect(image)

  if len(face_detection_result.detections) ==0:
     return 'no_face'
  elif len(face_detection_result.detections) > 1:
     return 'multiple_face'
  else:
    height, width, _ = img.shape
    key_point_name = ['right_eye', 'left_eye', 'nose', 'mouth', 'right_ear', 'left_ear']
    key_points = face_detection_result.detections[0].keypoints
    keypoint_px  = [_normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height) for keypoint in key_points]
    keypoint_dict = dict(zip(key_point_name, keypoint_px))
    pred_face_orientation = _face_orient_classifier(keypoint_dict)
    return pred_face_orientation


def main():
  directory = 'Data/IPhone/'
  pred_directory = 'Data/pred/iphone/'

  os.makedirs(pred_directory, exist_ok=True)
  key_point_name = ['class_gt', 'class_pred']

  data = {key: [] for key in key_point_name}
  for root, _, files in os.walk(directory):
      for file in files:
          if file.lower().endswith(('.jpg', '.png')):
              img_path = os.path.join(root, file)

              # Convert to PIL image to check and correct orientation
              img_pil = Image.open(img_path)
              img_pil = _correct_rotation(img_pil)
              img_pil = np.asarray(img_pil)

              face_detection_result = detect_face(img_pil)
              data['class_gt'].append(img_path.split('/')[-1])
              data['class_pred'].append(face_detection_result)

  df = pd.DataFrame(data)
  df.to_excel(pred_directory + "results.xlsx")

if __name__ == "__main__":
  main()