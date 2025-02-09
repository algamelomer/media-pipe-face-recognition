import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import queue
import speech_recognition as sr
import pyttsx3
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

CONFIDENCE_THRESHOLD = 0.2
ENROLLMENT_SHOTS = 5
ENROLLMENT_DELAY = 0.3
MIN_FACE_SIZE = 100
FRAMES_FOR_CONFIRMATION = 10

mp_face_detection = mp.solutions.face_detection
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
engine = pyttsx3.init()
recognizer = sr.Recognizer()

voice_queue = queue.Queue()

known_embeddings = []
known_labels = []
known_faces_dir = 'known_faces'
nn = None

def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()

def voice_worker():
    while True:
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=3)
                name = recognizer.recognize_google(audio)
                voice_queue.put(name)
        except Exception as e:
            voice_queue.put(None)

threading.Thread(target=voice_worker, daemon=True).start()

def get_embedding(face_image):
    face_image = cv2.resize(face_image, (224, 224))
    face_image = img_to_array(face_image)
    face_image = preprocess_input(face_image)
    return model.predict(np.expand_dims(face_image, axis=0))[0]

def load_known_faces():
    global known_embeddings, known_labels, nn
    known_embeddings = []
    known_labels = []
    
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
    
    for label in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, label)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                
                with mp_face_detection.FaceDetection(min_detection_confidence=0.85) as face_detector:
                    results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    if results.detections:
                        detection = results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        ih, iw, _ = image.shape
                        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                                    int(bbox.width * iw), int(bbox.height * ih)
                        
                        face = image[y:y+h, x:x+w]
                        try:
                            embedding = get_embedding(face)
                            known_embeddings.append(embedding)
                            known_labels.append(label)
                        except:
                            continue
    
    if len(known_embeddings) > 0:
        nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn.fit(known_embeddings)
    else:
        nn = None

class FaceEnroller:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.enrollment_active = False
        self.enrollment_shots = []
        self.confirmation_frames = 0
        self.last_capture_time = 0
        self.name = None
        self.progress = 0

    def update_progress(self, frame):
        self.progress = len(self.enrollment_shots)/ENROLLMENT_SHOTS
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        cv2.circle(frame, (cx, cy), 40, (255, 255, 255), 2)
        cv2.ellipse(frame, (cx, cy), (40, 40), -90, 0, 360*self.progress, 
                   (0, 255, 0), 4)
        cv2.putText(frame, f"{int(self.progress*100)}%", (cx-30, cy+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

load_known_faces()
face_enroller = FaceEnroller()

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.85) as face_detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        
        current_label = "Unknown"
        color = (0, 0, 255)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                        int(bbox.width * iw), int(bbox.height * ih)
            
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
                
            face = frame[y:y+h, x:x+w]
            
            try:
                embedding = get_embedding(face)
                
                if nn:
                    distance, idx = nn.kneighbors([embedding])
                    if distance[0][0] < CONFIDENCE_THRESHOLD:
                        current_label = known_labels[idx[0][0]]
                        color = (0, 255, 0)
                        face_enroller.reset()
                    else:
                        face_enroller.confirmation_frames += 1
                        if face_enroller.confirmation_frames > FRAMES_FOR_CONFIRMATION:
                            current_label = "Unknown"
                            color = (0, 0, 255)
                            if not face_enroller.enrollment_active:
                                face_enroller.enrollment_active = True
                                speak("New face detected. Please hold still for enrollment")
                        else:
                            current_label = "Verifying..."
                            color = (0, 165, 255)
                else:
                    current_label = "No known faces"
                    color = (0, 165, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, current_label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                if face_enroller.enrollment_active:
                    face_enroller.update_progress(frame)
                    
                    if (time.time() - face_enroller.last_capture_time > ENROLLMENT_DELAY and
                        len(face_enroller.enrollment_shots) < ENROLLMENT_SHOTS):
                        face_enroller.enrollment_shots.append(face)
                        face_enroller.last_capture_time = time.time()
                        
                    if len(face_enroller.enrollment_shots) >= ENROLLMENT_SHOTS:
                        if face_enroller.name is None:
                            speak("Please say your name")
                            try:
                                face_enroller.name = voice_queue.get(timeout=10)
                            except queue.Empty:
                                face_enroller.reset()
                                continue
                            
                        if face_enroller.name:
                            def save_and_retrain():
                                save_dir = os.path.join(known_faces_dir, face_enroller.name)
                                os.makedirs(save_dir, exist_ok=True)
                                for i, shot in enumerate(face_enroller.enrollment_shots):
                                    cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), shot)
                                load_known_faces()
                                speak(f"Welcome {face_enroller.name}!")
                            
                            threading.Thread(target=save_and_retrain).start()
                            face_enroller.reset()

            except Exception as e:
                print(f"Error: {e}")

        try:
            if not voice_queue.empty():
                voice_queue.get_nowait()
        except queue.Empty:
            pass

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()