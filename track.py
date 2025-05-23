#import osc_client
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker


# def print_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     # annotated_frame = draw_landmarks_on_image(output_image.numpy_view(), result)
#     # cv.imshow(':(', annotated_frame)
#     # if cv.waitKey(5) & 0xFF == ord("x"):
#     #     quit()
#
#     #print('{}'.format(result))
#     msg = client.modify_message(result)
#     client.send(msg)


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

VIDEO_STREAM_ADDRESS = "192.168.1.124:8080"
IP_DESTINATION = "127.0.0.1"
PORT_DESTINATION = 9001

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap0 = cv.VideoCapture(0)
cap1 = cv.VideoCapture('http://' + VIDEO_STREAM_ADDRESS + '/video')

keypoints_0 = []
keypoints_1 = []
keypoints_3d = []

#client = osc_client.OscClient(IP_DESTINATION, PORT_DESTINATION)

BASE_OPTIONS = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=BASE_OPTIONS,
   #output_segmentation_masks=True,
    running_mode=mp.tasks.vision.RunningMode.IMAGE)
    #result_callback=print_result)
detector = vision.PoseLandmarker.create_from_options(options)



timestamp0 = 0
timestamp1 = 0
with PoseLandmarker.create_from_options(options) as landmarker0, \
     PoseLandmarker.create_from_options(options) as landmarker1:

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not (ret0 & ret1):
            print("Ignoring empty frame")
            break

        timestamp0 += 1
        timestamp1 += 1

        mp_image_0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame0)
        mp_image_1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame1)

        result0 = landmarker0.detect(mp_image_0)
        result1 = landmarker1.detect(mp_image_1)

        pose_landmarks_list_0 = result0.pose_landmarks
        pose_landmarks_list_1 = result1.pose_landmarks
        pose_landmarks_proto_0 = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto_1 = landmark_pb2.NormalizedLandmarkList()

        for idx in range(len(pose_landmarks_list_0)):
          pose_landmarks = pose_landmarks_list_0[idx]

          pose_landmarks_proto_0 = landmark_pb2.NormalizedLandmarkList()
          pose_landmarks_proto_0.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
          ])

        for idx in range(len(pose_landmarks_list_1)):
          pose_landmarks = pose_landmarks_list_1[idx]

          pose_landmarks_proto_1 = landmark_pb2.NormalizedLandmarkList()
          pose_landmarks_proto_1.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
          ])

        img0 = draw_landmarks_on_image(frame0, result0)
        img1 = draw_landmarks_on_image(frame1, result1)

        cv.imshow('cam1', img1)
        cv.imshow('cam0', img0)
        if cv.waitKey(5) & 0xFF == ord("x"):
            quit()

cap0.release()
cap1.release()
cv.destroyAllWindows()