import cv2
import mediapipe as mp
import numpy as np

class FakeRealFaceDetection:
    def __init__(self, blur_threshold=35, confidence_threshold=0.8):
        self.blur_threshold = blur_threshold
        self.confidence_threshold = confidence_threshold
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.circle_center = None
        self.circle_radius = None
        self.cam_width = 640
        self.cam_height = 480

    def detect_fake_or_real(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return frame, "No face detected"

        for face_landmarks in results.multi_face_landmarks:
            # Calculate face bounding box
            ih, iw, _ = frame.shape
            x_min = int(min([lm.x for lm in face_landmarks.landmark]) * iw)
            y_min = int(min([lm.y for lm in face_landmarks.landmark]) * ih)
            x_max = int(max([lm.x for lm in face_landmarks.landmark]) * iw)
            y_max = int(max([lm.y for lm in face_landmarks.landmark]) * ih)

            face = frame[y_min:y_max, x_min:x_max]
            blur_value = self.calculate_blur(face)

            if blur_value > self.blur_threshold:
                label = "Real Face"
                color = (0, 255, 0)  # Green for real
            else:
                label = "Fake Face"
                color = (0, 0, 255)  # Red for fake

            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        return frame, label

    def calculate_blur(self, face_image):
        """Calculate the blurriness of the face image using Laplacian variance."""
        return cv2.Laplacian(face_image, cv2.CV_64F).var()
