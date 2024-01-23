from typing import Tuple, Union
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
import pandas as pd
import os
from PIL import Image, ExifTags

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, 
                                     image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
  """
    Converts normalized keypoint values (x, y) to pixel coordinates based on the image dimensions.

    Args:
        normalized_x (float): The normalized x-coordinate of the keypoint (value between 0.0 and 1.0).
        normalized_y (float): The normalized y-coordinate of the keypoint (value between 0.0 and 1.0).
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.

    Returns:
        Union[None, Tuple[int, int]]: A tuple (x, y) representing the pixel coordinates of the keypoint.
                                      Returns None if the normalized coordinates are outside the [0.0, 1.0] range.
    """

  if not (0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0):
     return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def _get_keypoints(face_detection_result, img_shape: Tuple[int, int, int]) -> dict:
   """
    Normalizes the keypoints from a face detection result and maps them to their respective face attributes.

    Args:
        face_detection_result: The result of a face detection process, expected to contain data from which 
                               keypoints can be extracted (e.g., facial landmarks or features).
        img_shape (tuple): The shape of the image on which detection was performed, typically in the 
                           form (height, width, channels).

    Returns:
        dict: A dictionary mapping face attribute names (e.g., 'left_eye', 'right_eye') to their 
              respective pixel coordinates in the image (x, y).
    """
   height, width, _ = img_shape
   face_attribute_name = ['right_eye', 'left_eye', 'nose', 'mouth', 'right_ear', 'left_ear']
   key_points = face_detection_result.keypoints
   keypoint_px  = [_normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height) for keypoint in key_points]
   keypoint_dict = dict(zip(face_attribute_name, keypoint_px))
   return keypoint_dict


def _visualize(image: np.array, detection_result: list) -> np.array:
  """
  Draws bounding boxes and keypoints on the input image based on detection results and returns the modified image.

  Args:
      image (np.array): An RGB image represented as a NumPy array.
      detection_result (list): A list of 'Detection' entities. Each 'Detection' entity should contain 
                                information necessary for visualization, like coordinates for bounding boxes 
                                and keypoints.

  Returns:
      np.array: The input image array with bounding boxes and keypoints drawn on it.
  """
  annotated_image = image.copy()

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
    keypoint_dict = _get_keypoints(detection, image.shape)

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


def _classify_face_orientation(keypoints: dict) -> str:
  """
  Classifies the orientation of a face (frontal or profile) based on facial keypoints.

  Args:
      keypoints (dict): A dictionary containing facial keypoints. Each key represents a facial 
                        feature (e.g., 'left_eye', 'right_eye'), and its value is a tuple 
                        indicating the pixel coordinates (x, y) of the feature.

  Returns:
      str: A string indicating the face orientation. Returns 'front' for a frontal face 
            orientation and 'profile' for a profile face orientation.
  """
  angle_left_eye = math.degrees(math.atan2(keypoints['nose'][1] - keypoints['left_eye'][1], keypoints['nose'][0] - keypoints['left_eye'][0]))
  angle_right_eye = math.degrees(math.atan2(keypoints['nose'][1] - keypoints['right_eye'][1], keypoints['nose'][0] - keypoints['right_eye'][0]))

  if angle_left_eye > 90 > angle_right_eye :
    prediction = 'front'
  elif (angle_left_eye >= 90 and angle_right_eye >= 90) or (angle_left_eye <= 90 and angle_right_eye <= 90):
    prediction = 'profile'

  return prediction


def _correct_image_rotation(img: Image) -> Image:
  """
  Corrects the orientation of an image based on its EXIF data.

  This function checks for the 'Orientation' tag in the image's EXIF data and rotates the image accordingly.
  If the EXIF data is missing or does not contain the orientation tag, the image is returned as is.

  Args:
      img (Image): An instance of PIL.Image that potentially contains EXIF orientation data.

  Returns:
      Image: The image, potentially rotated to the correct orientation.
  """
  try:
      if hasattr(img, '_getexif'): # only present in PIL images
          for orientation in ExifTags.TAGS.keys():
              if ExifTags.TAGS[orientation] == 'Orientation':
                  break
          exif_data = img._getexif()
          if exif_data is not None:
              exif = dict(exif_data.items())
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


def classify_detection(face_detection_result: mp.tasks.vision.FaceDetectorResult, img_shape: Tuple[int, int, int]) -> str:
  """
    Classifies the detection result from a face detector into categories based on the number of faces detected 
    and their orientations.

    Args:
        face_detection_result (mp.solutions.face_detection.FaceDetectionResult): The result object from the 
            mediapipe face detection process.
        img_shape (Tuple[int, int, int]): The shape of the image as a tuple (height, width, channels).

    Returns:
        str: A string classification of the detection result. Possible values are 'no_face', 'multiple_faces', 
             or the face orientation ('front', 'profile', etc.) as determined by `_classify_face_orientation`.
    """
  if len(face_detection_result.detections) == 0:
     return 'no_face'
  elif len(face_detection_result.detections) > 1:
     return 'multiple_face'
  else:
    keypoint_dict = _get_keypoints(face_detection_result.detections[0], img_shape)
    pred_face_orientation = _classify_face_orientation(keypoint_dict)
    return pred_face_orientation


def detect_face(img: np.array) -> mp.tasks.vision.FaceDetectorResult:
  """
    Detects faces in an image using mediapipe's face detection model.

    Args:
        img (np.array): An image represented as a NumPy array in which faces are to be detected.
                        The image should be in RGB format.

    Returns:
        mp.solutions.face_detection.FaceDetectorResult: The result of the face detection process,
                                                        containing detected face information.
    """
  base_options = python.BaseOptions(model_asset_path='detector.tflite')
  options = python.vision.FaceDetectorOptions(base_options=base_options)
  detector = python.vision.FaceDetector.create_from_options(options)
  image = mp.Image(data=np.asarray(img), image_format=mp.ImageFormat.SRGB)

  face_detection_result = detector.detect(image)

  return face_detection_result


def predict(img: np.ndarray)->str:
   """
    Predicts the face (no_face, multiple_face, frontal, profile) in a given image.

    Args:
        img (np.array): A 3-dimensional NumPy array representing an image.

    Returns:
        str: A string prediction of the face orientation. Possible values include 'no_face', 
             'multiple_faces', 'front', 'profile', etc., based on the detection and classification 
             results.
    """
   detection_result = detect_face(img)
   prediction = classify_detection(detection_result, img.shape)
   return prediction


def main():
  # Example usage
  data_directory = 'Data/IPhone/'
  pred_directory = 'Data/pred/iphone/'

  os.makedirs(pred_directory, exist_ok=True)
  key_point_name = ['class_gt', 'class_pred']

  data = {key: [] for key in key_point_name}
  for root, _, files in os.walk(data_directory):
      for file in files:
          if file.lower().endswith(('.jpg', '.png')):
              img_path = os.path.join(root, file)

              # Convert to PIL image to check and correct orientation
              img_pil = Image.open(img_path)
              img_pil = _correct_image_rotation(img_pil)
              img = np.asarray(img_pil)

              detection_result = detect_face(img_pil)
              prediction = classify_detection(detection_result, img.shape)
              # annotated_image = _visualize(img, detection_result)
              # annotated_image = Image.fromarray(annotated_image)
              # annotated_image.save(pred_directory + file)
              data['class_gt'].append(img_path.split('/')[-1])
              data['class_pred'].append(prediction)

  df = pd.DataFrame(data)
  df.to_excel(pred_directory + "results.xlsx")

if __name__ == "__main__":
  main()