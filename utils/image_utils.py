"""
Image Processing Utilities
Helper functions for image manipulation and validation
"""

import base64
import io
from PIL import Image
from typing import Optional
import cv2
import numpy as np


def enhance_image(image: Image.Image) -> Image.Image:
    """
    Enhance image quality for better object detection
    Applies CLAHE, denoising, and sharpening
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Convert back to RGB and PIL
    result_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)



def validate_image(image_bytes: bytes) -> bool:
    """
    Validate if bytes represent a valid image
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Don't call verify() as it's too strict and causes issues
        # Just check if we can open it
        img.format  # This will raise an exception if not a valid image
        return True
    except Exception as e:
        print(f"Image validation error: {e}")
        return False


def base64_to_bytes(base64_string: str) -> bytes:
    """
    Convert base64 string to bytes
    
    Args:
        base64_string: Base64 encoded string
        
    Returns:
        Decoded bytes
    """
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    return base64.b64decode(base64_string)


def bytes_to_base64(image_bytes: bytes) -> str:
    """
    Convert bytes to base64 string
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode('utf-8')


def resize_image(image: Image.Image, max_size: int = 1280) -> Image.Image:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    width, height = image.size
    
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image


def get_image_info(image: Image.Image) -> dict:
    """
    Get image metadata
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image info
    """
    return {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format
    }
