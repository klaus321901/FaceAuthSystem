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
```

---

## How It Works

1. Employee face images are stored locally inside the `employees/` directory.

2. During startup, the system generates embeddings for all employee images using FaceNet.

3. When a new image is uploaded:
   - RetinaFace detects faces
   - FaceNet generates embeddings
   - Cosine similarity compares embeddings with the employee database

4. The system returns:
   - Recognized employee name
   - Access status
   - Confidence score
   - Annotated image with bounding boxes

---
## Installation

Clone the repository:

```bash
git clone https://github.com/klaus321901/FaceAuthSystem.git
cd FaceAuthSystem
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment (Windows):

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python main.py
```

Server starts at:

```text
http://localhost:8000
```

---

## API Endpoints

### 1. Recognize Faces

```http
POST /api/recognize
```

Upload an image and receive recognition results.

### Sample Response

```json
{
  "success": true,
  "detections": [
    {
      "name": "John Doe",
      "access": "granted",
      "confidence": 0.91
    }
  ],
  "count": 1,
  "message": "Detected 1 face(s): 1 granted, 0 denied"
}
```

---

### 2. Health Check

```http
GET /api/health
```

Returns:
- System status
- Registered employee count
- Model information

---

### 3. Employee List

```http
GET /api/employees
```

Returns all registered employees and image counts.

---

## Recognition Pipeline

### Face Detection
- RetinaFace detector backend

### Face Embedding Model
- FaceNet

### Similarity Metric
- Cosine Distance

### Recognition Threshold

```python
threshold = 0.55
```

---

## Security & Privacy

The `employees/` folder is excluded from GitHub using `.gitignore` because it contains private employee face images.

---

## Future Improvements

- Real-time webcam authentication
- JWT-based secure login
- Database integration
- Face registration API
- Liveness detection
- Docker deployment
- GPU acceleration
- Authentication logs

---

## Requirements

```text
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
deepface==0.0.75
opencv-python==4.8.1.78
tensorflow==2.12.0
pillow==10.1.0
numpy==1.24.3
tf-keras==2.13.0
```

---
