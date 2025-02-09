import cv2
from face_recognition import FaceRecognition
from FaceEnrollment import FaceEnrollment

def start_face_enrollment():
    """Start the face enrollment process."""
    user_name = input("Enter your name for enrollment: ")
    face_enrollment = FaceEnrollment(user_name)
    face_enrollment.enroll_face()
    print(f"Enrollment for {user_name} complete.")

def start_face_recognition():
    face_recognition = FaceRecognition()

    cap = cv2.VideoCapture(0)  # Use the first camera (change if needed)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face recognition
        frame = face_recognition.recognize_face(frame)

        # Show the frame with results
        cv2.imshow("Face Recognition", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    choice = input("Choose an option:\n1. Start Face Enrollment\n2. Start Face Recognition\nEnter choice (1/2): ")

    if choice == '1':
        start_face_enrollment()
    elif choice == '2':
        start_face_recognition()
    else:
        print("Invalid choice. Exiting.")
