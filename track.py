import osc_client
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker
import threading


result_lock = threading.Lock()
latest_result = None

def save_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result

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

# VIDEO_STREAM_ADDRESS = "192.168.1.124:8080"
IP_DESTINATION = "127.0.0.1"
PORT_DESTINATION = 9001

cap0 = cv.VideoCapture(0)

client = osc_client.OscClient(IP_DESTINATION, PORT_DESTINATION)

BASE_OPTIONS = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=BASE_OPTIONS,
   #output_segmentation_masks=True,
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=save_result)

timestamp0 = 0

with PoseLandmarker.create_from_options(options) as landmarker0:
    while True:
        ret0, frame0 = cap0.read()
        if not ret0:
            print("Ignoring empty frame")
            break

        timestamp0 += 1

        frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        mp_image_0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame0_rgb)

        landmarker0.detect_async(mp_image_0, timestamp0)

        annotated_frame = frame0.copy()

        with result_lock:
            if latest_result:
                annotated_frame = draw_landmarks_on_image(frame0_rgb, latest_result)
                annotated_frame = cv.cvtColor(annotated_frame, cv.COLOR_RGB2BGR)

                osc_message = client.modify_message(latest_result)
                client.send(osc_message)

                latest_result = None

        cv.imshow(':(', annotated_frame)

        if cv.waitKey(5) & 0xFF == ord('x'):
            break

cap0.release()
cv.destroyAllWindows()