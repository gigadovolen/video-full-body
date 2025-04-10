from numpy.matrixlib.defmatrix import matrix
from pythonosc import udp_client
import mediapipe as mp
from mediapipe.tasks import python
import numpy as np
import math
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker

class Vector3:
    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"

    def __str__(self) -> str:
        return f"({self.x},{self.y},{self.z})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector3):
            raise NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __add__(self, other) -> "Vector3":
        if not isinstance(other, Vector3):
            raise NotImplemented
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if not isinstance(other, Vector3):
            raise NotImplemented
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=dtype)

    def convert_hand(self):
        self.z = -self.z

    def scale(self, xscale = 1.0, yscale = 1.0, zscale = 1.0):
        self.x *= xscale
        self.y *= yscale
        self.z *= zscale

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def length(self) -> np.floating:
        return np.linalg.norm(self)

    def normalize(self):
        norm = np.linalg.norm(self)
        if norm == 0:
            return self
        return self / norm

class RotationSolver:
    def __init__(self, v1: Vector3, target: Vector3):
        self.v1 = v1
        self.target = target

    def rotation_mat(self) -> matrix:
        axis_z = Vector3.normalize(self.v1 - self.target)
        if Vector3.length(axis_z) == 0:
            axis_z = np.array((0, -1, 0))

        axis_x = np.cross(np.array((0, 0, 1)), axis_z)
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array((1, 0, 0))

        axis_y = np.cross(axis_z, axis_x)
        rot_matrix = np.matrix([axis_x, axis_y, axis_z]).transpose()
        return rot_matrix

    @staticmethod
    def euler(R: matrix) -> Vector3:
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        locked = sy < 1e-6

        if not locked:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return Vector3(x, y, z)

class OscClient:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

        try:
            self.client = udp_client.SimpleUDPClient(ip, port)
            print("OSC client started")
        except Exception as e:
            print("Unable to start OSC client: ", e)

    @staticmethod
    def modify_message(input: mp.tasks.vision.PoseLandmarkerResult) -> dict[str, tuple[Vector3, Vector3]] | bool:
            if input.pose_world_landmarks:
                landmarks = input.pose_world_landmarks[0]
                frame = []
                try:
                    for landmark in landmarks:                                                           #https://ai.google.dev/static/edge/mediapipe/images/solutions/pose_landmarks_index.png
                        frame.append(Vector3(float(landmark.x), float(landmark.y), -float(landmark.z)))  #list c 33 туплями
                except Exception as e:
                    print(f"Frame failed: {e}")

                ret = {"1": ((frame[22] + frame[23]).scale(0.5, 0.5, 0.5), Vector3()),
                       "2": ((frame[10] + frame[11]).scale(0.5, 0.5, 0.5), Vector3()),
                       "3": (frame[28], Vector3()),
                       "4": (frame[29], Vector3()),             #1-hip,2-chest,34-feet,56-knees,78-elbows
                       "5": (frame[24], Vector3()),
                       "6": (frame[25], Vector3()),
                       "7": (frame[12], Vector3()),
                       "8": (frame[13], Vector3()),
                       "head": (frame[0], Vector3())}

                # try:
                #     head_mat = RotationSolver(ret.get("head")[0], ret.get("1")[0]).rotation_mat()
                #     head_rot = RotationSolver.euler(head_mat)
                #     elbow1_mat = RotationSolver(ret.get("12")[0], ret.get("14")[0]).rotation_mat()
                #     elbow1_rot = RotationSolver.euler(elbow1_mat)
                #     elbow2_mat = RotationSolver(ret.get("13")[0], ret.get("15")[0]).rotation_mat()
                #     elbow2_rot = RotationSolver.euler(elbow2_mat)
                #     knee1_mat = RotationSolver(ret.get("24")[0], ret.get("26")[0]).rotation_mat()
                #     knee1_rot = RotationSolver.euler(knee1_mat)
                #     knee2_mat = RotationSolver(ret.get("25")[0], ret.get("27")[0]).rotation_mat()
                #     knee2_rot = RotationSolver.euler(knee2_mat)
                #
                #     ret.update({"head": (frame[0], head_rot)})
                #     ret.update({"7": (frame[0], elbow1_rot)})
                #     ret.update({"8": (frame[0], elbow2_rot)})
                #     ret.update({"5": (frame[0], knee1_rot)})
                #     ret.update({"6": (frame[0], knee2_rot)})
                # except Exception as e:
                #     print(f"Rotation failed horribly: {e}")

                return ret
            return False

    def send(self, frame: dict[str, tuple[Vector3, Vector3]] | bool):
        if isinstance(frame, bool):
            print("Skipped frame")
            pass

        else:
            for addr, message in frame.items():
                print("Message is almost ready")

                position = message[0]
                rotation = message[1]
                address = f"/tracking/trackers/{addr}/position"
                address_rot = f"/tracking/trackers/{addr}/rotation"
                print("Message is ready")

                if isinstance(message, Vector3):
                    try:
                        position_vector = position.to_tuple()
                        rotation_vector = rotation.to_tuple()

                        self.client.send_message(address, position_vector)
                        print(f"Sent message {position_vector}")
                        self.client.send_message(address_rot, rotation_vector)
                        print(f"Sent message {rotation_vector}")
                    except Exception as e:
                        print(f"Failed to send message: {e}")