import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker


def print_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    annotated_frame = draw_landmarks_on_image(output_image.numpy_view(), result)
    cv.imshow(':(', annotated_frame)
    print('{}'.format(result))

def draw_landmarks_on_image(frame, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(frame)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

stream = cv.VideoCapture('http://192.168.1.97:8080/video')
if not stream.isOpened():
    print("Cannot open stream")
    exit()

base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
   # output_segmentation_masks=True,
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=print_result)
detector = vision.PoseLandmarker.create_from_options(options)


timestamp = 0
with PoseLandmarker.create_from_options(options) as landmarker:

    while stream.isOpened():
        # Capture frame-by-frame
        ret, frame = stream.read()

        if not ret:
            print("Ignoring empty frame")
            break

        cv.imshow(':(', frame)
        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)

        if cv.waitKey(5) & 0xFF == ord("x"):
            break

stream.release()
cv.destroyAllWindows()
cv.waitKey(5)
