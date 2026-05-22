"""
API Routes for Face Recognition Authentication System
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import logging

from face_recognizer import get_recognizer

router = APIRouter(prefix="/api", tags=["face-recognition"])
logger = logging.getLogger(__name__)

class RecognitionResponse(BaseModel):
    success: bool
    detections: List[Dict]
    image: str
    count: int
    message: str = ""

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_faces(file: UploadFile = File(...)):
    """
    Recognize faces in uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Recognition results with annotated image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read file bytes
        image_bytes = await file.read()
        
        # Get recognizer and process image
        recognizer = get_recognizer()
        detections, annotated_image_base64 = recognizer.process_image_bytes(image_bytes)
        
        # Count access granted vs denied
        granted_count = sum(1 for d in detections if d['access'] == 'granted')
        denied_count = sum(1 for d in detections if d['access'] == 'denied')
        
        message = f"Detected {len(detections)} face(s): {granted_count} granted, {denied_count} denied"
        
        logger.info(message)
        
        return RecognitionResponse(
            success=True,
            detections=detections,
            image=annotated_image_base64,
            count=len(detections),
            message=message
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    recognizer = get_recognizer()
    employee_count = len(recognizer.employee_database)
    
    return {
        "status": "healthy",
        "employees_registered": employee_count,
        "model": recognizer.model_name,
        "detector": recognizer.detector_backend
    }

@router.get("/employees")
async def list_employees():
    """List all registered employees"""
    recognizer = get_recognizer()
    
    employees = []
    for name, embeddings in recognizer.employee_database.items():
        employees.append({
            "name": name.replace('_', ' '),
            "images_count": len(embeddings)
        })
    
    return {
        "total": len(employees),
        "employees": employees
    }
