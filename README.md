# MediaPipe Face Recognition

## Overview
This project implements a Face ID-style face enrollment and recognition system using OpenCV and MediaPipe. It features an object-oriented structure and provides real-time face alignment guidance with animations.

## Features
- **Face Enrollment System:** Users can enroll their faces with a given name.
- **Face Mesh Visualization:** Displays face mesh dots for a modern look.
- **Face Alignment Guidance:** A dynamic alignment circle changes color:
  - **Green:** Properly aligned face.
  - **Red:** Misaligned face.
  - **Gray:** No face detected.
- **Gesture-Based Movement Guidance:** Users receive visual cues to move their heads for better face coverage.
- **Modular Code Structure:** Organized into different files for handling streaming, controls, and UI elements.

## Installation
### Requirements
Ensure you have Python 3.12 installed. Install the required dependencies using:
```sh
pip install -r requirements.txt
```

### Dependencies
- OpenCV
- MediaPipe
- Flask (for potential integration)
- Pygame (for UI interactions, if applicable)

## Usage
1. Run the face enrollment script:
   ```sh
   python enroll.py
   ```
2. Follow on-screen instructions for face alignment.
3. Once enrolled, run face recognition:
   ```sh
   python recognize.py
   ```
4. The system will detect and identify faces in real-time.

## Project Structure
```
📂 media-pipe-face-recognition
├── 📂 src
│   ├── stream_handler.py  # Handles video streaming
│   ├── face_recognition.py  # Main face recognition logic
│   ├── face_enrollment.py  # Enrollment system
│   ├── ui.py  # Pygame-based UI elements
│   └── utils.py  # Utility functions
├── requirements.txt  # Dependencies
├── enroll.py  # Script for enrolling new faces
├── recognize.py  # Script for recognizing faces
├── README.md  # Project documentation
```

## Future Enhancements
- Improve face recognition accuracy with deep learning models.
- Implement a database for storing enrolled users.
- Integrate with a web or mobile app for user management.

## Author
**Omar Algamel**

## License
This project is open-source and available under the MIT License.

