import cv2
import numpy as np

class FaceDetector:
    def __init__(self, backend="HC"):
        self.backend = backend
        if backend == "HC":
            # Use OpenCV's Haar Cascade algorithm implementation for face detection
            # This should only utilize the CPU
            self.detector = cv2.CascadeClassifier('./dataset/haarcascade_frontalface_default.xml')
        elif backend == "RF":
            from retinaface import RetinaFace
            self.detector = RetinaFace
        elif "Y5F" in backend:
            from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
            self.detector = YOLO5Face()

        else:
            raise ValueError("Unsupported face detection backend!")

    def detect(self, frame, use_larger_box=False, larger_box_coef=1.0):
        if self.backend == "HC":
            # Computed face_zone(s) are in the form [x_coord, y_coord, width, height]
            # (x,y) corresponds to the top-left corner of the zone to define using
            # the computed width and height.
            face_zone = self.detector.detectMultiScale(frame[:, :, :3].astype(np.uint8))

            if len(face_zone) < 1:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
            elif len(face_zone) >= 2:
                # Find the index of the largest face zone
                # The face zones are boxes, so the width and height are the same
                max_width_index = np.argmax(face_zone[:, 2])  # Index of maximum width
                face_box_coor = face_zone[max_width_index]
                print("Warning: More than one faces are detected. Only cropping the biggest one.")
            else:
                face_box_coor = face_zone[0]
        elif self.backend == "RF":
            # Use a TensorFlow-based RetinaFace implementation for face detection
            # This utilizes both the CPU and GPU
            res = self.detector.detect_faces(frame[:, :, :3].astype(np.uint8))

            if len(res) > 0:
                # Pick the highest score
                highest_score_face = max(res.values(), key=lambda x: x['score'])
                face_zone = highest_score_face['facial_area']

                # This implementation of RetinaFace returns a face_zone in the
                # form [x_min, y_min, x_max, y_max] that corresponds to the
                # corners of a face zone
                x_min, y_min, x_max, y_max = face_zone

                # Convert to this toolbox's expected format
                # Expected format: [x_coord, y_coord, width, height]
                x = x_min
                y = y_min
                width = x_max - x_min
                height = y_max - y_min

                # Find the center of the face zone
                center_x = x + width // 2
                center_y = y + height // 2

                # Determine the size of the square (use the maximum of width and height)
                square_size = max(width, height)

                # Calculate the new coordinates for a square face zone
                new_x = center_x - (square_size // 2)
                new_y = center_y - (square_size // 2)
                face_box_coor = [new_x, new_y, square_size, square_size]
            else:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        elif "Y5F" in self.backend :
            res = self.detector.detect_face(frame[:, :, :3].astype(np.uint8))
            if res != None:
                x_min, y_min, x_max, y_max = res

                # Convert to this toolbox's expected format
                # Expected format: [x_coord, y_coord, width, height]
                x = x_min
                y = y_min
                width = x_max - x_min
                height = y_max - y_min

                # Find the center of the face zone
                center_x = x + width // 2
                center_y = y + height // 2

                # Determine the size of the square (use the maximum of width and height)
                square_size = max(width, height)

                # Calculate the new coordinates for a square face zone
                new_x = center_x - (square_size // 2)
                new_y = center_y - (square_size // 2)
                face_box_coor = [new_x, new_y, square_size, square_size]

            else:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        return face_box_coor


# Resize back to original size
def resize_to_original(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def face_detection(frame, detector, use_larger_box=False, larger_box_coef=1.0):
    """Face detection on a single frame.

    Args:
        frame(np.array): a single frame.
        detector(class): backend to utilize for face detection.
        use_larger_box(bool): whether to use a larger bounding box on face detection.
        larger_box_coef(float): Coef. of larger box.
    Returns:
        face_box_coor(List[int]): coordinates of face bouding box.
    """
    return detector.detect(frame, use_larger_box, larger_box_coef)
