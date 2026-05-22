# Face Recognition Authentication System

A FastAPI-based face recognition authentication system that identifies employees from uploaded images using DeepFace, FaceNet embeddings, and RetinaFace detection.

The system detects faces in an image, compares them against a registered employee database, and returns authentication results with annotated bounding boxes.

---

## Features

- Face recognition based employee authentication
- FastAPI backend with REST API endpoints
- RetinaFace for accurate face detection
- FaceNet embeddings for face recognition
- Cosine similarity matching
- Real-time image annotation
- Access granted / denied classification
- Employee database loading from local image folders
- Static frontend using HTML, CSS, and JavaScript
- Health monitoring endpoint

---

## Tech Stack

### Backend
- Python
- FastAPI
- Uvicorn

### Face Recognition & AI
- DeepFace
- FaceNet
- RetinaFace
- TensorFlow
- OpenCV

### Frontend
- HTML
- CSS
- JavaScript

---

## Project Structure

```text
FaceAuthSystem/
│
├── api/
│   └── routes.py
│
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── utils/
│   ├── demo_helper.py
│   ├── image_utils.py
│   └── validation.py
│
├── employees/              # Ignored from GitHub (private employee images)
│
├── face_recognizer.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
---

## How It Works
Employee face images are stored locally inside the employees/ directory.
During startup, the system generates embeddings for all employee images using FaceNet.
When a new image is uploaded:
RetinaFace detects faces
FaceNet generates embeddings
Cosine similarity compares embeddings with the employee database
The system returns:
Recognized employee name
Access status
Confidence score
Annotated image with bounding boxes
