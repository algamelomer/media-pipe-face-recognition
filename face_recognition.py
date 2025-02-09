import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

class FaceRecognition:
    def __init__(self, enrolled_faces_dir='enrolled_faces', threshold=0.15):
        self.enrolled_faces_dir = enrolled_faces_dir
        self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.95)
        self.known_embeddings = []
        self.known_labels = []
        self.nn = None
        self.threshold = threshold  # Cosine distance threshold for "real" vs "fake"
        self.load_known_faces()

    def get_embedding(self, face_image):
        """Convert face image to embedding vector"""
        face_image = cv2.resize(face_image, (224, 224))
        face_image = img_to_array(face_image)
        face_image = preprocess_input(face_image)
        embedding = self.model.predict(np.expand_dims(face_image, axis=0))[0]
        return embedding

    def load_known_faces(self):
        """Load enrolled faces and generate embeddings"""
        for label in os.listdir(self.enrolled_faces_dir):
            person_dir = os.path.join(self.enrolled_faces_dir, label)
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)

                results = self.face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                    face = image[y:y+h, x:x+w]
                    embedding = self.get_embedding(face)
                    self.known_embeddings.append(embedding)
                    self.known_labels.append(label)

        self.nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.nn.fit(self.known_embeddings)

    def recognize_face(self, frame):
        """Detect face in the frame and recognize it, also check for "real" vs "fake" face"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)

        if not results.detections:
            return frame  # No face detected, just return the original frame

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue  # Skip if no face region is found

            # Get the embedding of the detected face
            embedding = self.get_embedding(face)

            # Compare with known faces
            distances, indices = self.nn.kneighbors([embedding])
            cosine_distance = distances[0][0]

            # Check if the distance is below a certain threshold
            if cosine_distance < self.threshold:
                label = self.known_labels[indices[0][0]]
                result = f"{label} - Real"
                color = (0, 255, 0)  # Green for real
            elif cosine_distance < self.threshold * 2:
                label = self.known_labels[indices[0][0]]
                result = f"{label} - Fake"
                color = (0, 0, 255)  # Red for fake
            else:
                result = "Unknown"
                color = (128, 128, 128)  # Gray for unknown

            # Draw a bounding box, label, and the "real", "fake", or "unknown" text
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame
