import cv2
import mediapipe as mp
import numpy as np
import os
import time

class FaceEnrollment:
    def __init__(self, user_name, save_path="enrolled_faces"):
        self.user_name = user_name
        self.save_path = save_path
        self.circle_center = None
        self.circle_radius = None
        self.saved_images = 0
        self.max_images = 10  # Number of images required for enrollment

        # Initialize Mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

        # Create directory if it doesn't exist
        os.makedirs(os.path.join(self.save_path, self.user_name), exist_ok=True)

        # Open Webcam
        self.cap = cv2.VideoCapture(0)
        self.screen_width = int(self.cap.get(3))
        self.screen_height = int(self.cap.get(4))

        # Define Circle Position
        self.circle_center = (self.screen_width // 2, self.screen_height // 2 )
        self.circle_radius = min(self.screen_width, self.screen_height) // 3

    def enroll_face(self):
        delay_time = time.time()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face Detection
            results = self.face_mesh.process(rgb_frame)
            
            face_inside = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get nose tip position
                    nose_x = int(face_landmarks.landmark[1].x * self.screen_width)
                    nose_y = int(face_landmarks.landmark[1].y * self.screen_height)
                    face_center = (nose_x, nose_y)

                    # Check if nose is inside the circle
                    if np.linalg.norm(np.array(self.circle_center) - np.array(face_center)) < self.circle_radius:
                        face_inside = True
                        if time.time() - delay_time > 1:  # Capture image every second
                            self.save_face(frame)
                            delay_time = time.time()

                    # Draw face mesh points for a modern look
                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * self.screen_width), int(lm.y * self.screen_height)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Change circle color based on face alignment
            if not results.multi_face_landmarks:
                circle_color = (100, 100, 100)  # Gray (no face detected)
            elif face_inside:
                circle_color = (0, 255, 0)  # Green (aligned correctly)
            else:
                circle_color = (0, 0, 255)  # Red (misaligned)

            # Draw radial lines around the circle
            num_lines = 72  # Increased number of segments
            for i in range(num_lines):
                angle = i * (360 / num_lines)
                x1 = int(self.circle_center[0] + (self.circle_radius + 20) * np.cos(np.radians(angle)))
                y1 = int(self.circle_center[1] + (self.circle_radius + 20) * np.sin(np.radians(angle)))
                x2 = int(self.circle_center[0] + (self.circle_radius + 30) * np.cos(np.radians(angle)))
                y2 = int(self.circle_center[1] + (self.circle_radius + 30) * np.sin(np.radians(angle)))
                cv2.line(frame, (x1, y1), (x2, y2), circle_color, 2)

            # Draw Face ID Circle
            cv2.circle(frame, self.circle_center, self.circle_radius + 10, circle_color, 10, cv2.LINE_AA)
            
            # Show completion message
            if self.saved_images >= self.max_images:
                cv2.putText(frame, "Face Enrollment Complete!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow("Face Enrollment", frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or self.saved_images >= self.max_images:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def save_face(self, frame):
        """Save the captured face image."""
        if self.saved_images < self.max_images:
            img_path = os.path.join(self.save_path, self.user_name, f"{self.saved_images}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path}")
            self.saved_images += 1
