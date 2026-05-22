"""
Face Recognition System for Data Center Authentication
Compatible with Latest DeepFace + RetinaFace
"""
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import os
import io
import base64
from typing import List, Dict, Tuple
class FaceRecognizer:
    def __init__(self, employee_db_path="employees"):
        self.employee_db_path = employee_db_path
        self.employee_database = {}
        # ✅ RetinaFace for accurate detection
        self.model_name = "Facenet"
        self.detector_backend = "retinaface"
        self.distance_metric = "cosine"
        self.threshold = 0.55
        print("=" * 60)
        print("🔐 Initializing Face Recognition System")
        print("=" * 60)
        self.load_employee_database()
    def load_employee_database(self):
        if not os.path.exists(self.employee_db_path):
            print("⚠️ Employee folder not found.")
            return
        print(f"📂 Loading employee database from: {self.employee_db_path}")
        for person in os.listdir(self.employee_db_path):
            person_dir = os.path.join(self.employee_db_path, person)
            if not os.path.isdir(person_dir):
                continue
            embeddings = []
            for img in os.listdir(person_dir):
                if not img.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                img_path = os.path.join(person_dir, img)
                try:
                    result = DeepFace.represent(
                        img_path=img_path,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False
                    )
                    if result and isinstance(result, list) and len(result) > 0:
                        embeddings.append(result[0]['embedding'])
                        print(f"   ✅ Loaded {img}")
                except Exception as e:
                    print(f"   ❌ Failed {img}: {e}")
            if embeddings:
                self.employee_database[person] = embeddings
        print("=" * 60)
        print(f"✅ Database loaded: {len(self.employee_database)} employee(s)")
        print("=" * 60)
    def recognize(self, image: Image.Image):
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        results = []
        try:
            # Use RetinaFace to detect faces (more accurate, no false positives)
            faces = DeepFace.extract_faces(
                img_path=img_np,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            print(f"Detected {len(faces)} face(s) using RetinaFace")
            
            for face_obj in faces:
                facial_area = face_obj.get('facial_area', {})
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)
                
                face_img = face_obj.get('face')
                
                if face_img is not None:
                    try:
                        # Convert normalized face to uint8
                        if face_img.max() <= 1.0:
                            face_img = (face_img * 255).astype(np.uint8)
                        
                        # Get embedding
                        emb_result = DeepFace.represent(
                            img_path=face_img,
                            model_name=self.model_name,
                            detector_backend=self.detector_backend,
                            enforce_detection=False
                        )
                        if emb_result and isinstance(emb_result, list) and len(emb_result) > 0:
                            emb = emb_result[0]['embedding']
                            
                            # Match against database
                            name, access, conf = self.match_embedding(emb)
                            results.append({
                                'name': name,
                                'access': access,
                                'confidence': conf
                            })
                            # Draw bounding box
                            color = (0, 255, 0) if access == 'granted' else (0, 0, 255)
                            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 3)
                            # Draw label background
                            label_height = 80
                            cv2.rectangle(img_bgr, (x, y - label_height), (x + w, y), color, -1)
                            # Draw text
                            label = name
                            status = "ACCESS GRANTED" if access == 'granted' else "ACCESS DENIED"
                            conf_text = f"Confidence: {conf*100:.1f}%" if conf > 0 else ""
                            cv2.putText(img_bgr, label, (x + 5, y - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(img_bgr, status, (x + 5, y - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            if conf_text:
                                cv2.putText(img_bgr, conf_text, (x + 5, y - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error detecting faces: {e}")
        
        img_annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return results, img_annotated
    def match_embedding(self, emb):
        best_dist = 999
        best_name = "Unknown"
        for name, db_embs in self.employee_database.items():
            for db_emb in db_embs:
                d = self.cosine_distance(emb, db_emb)
                if d < best_dist:
                    best_dist = d
                    best_name = name
        if best_dist < self.threshold:
            return best_name.replace("_", " "), "granted", round(1 - best_dist, 2)
        return "Unknown Person", "denied", 0.0
    def cosine_distance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    def process_image_bytes(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        detections, annotated = self.recognize(image)
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        return detections, img_b64
_recognizer = None
def get_recognizer() -> FaceRecognizer:
    global _recognizer
    _recognizer = FaceRecognizer()
    return _recognizer