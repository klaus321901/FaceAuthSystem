"""
Demo Mode Helper
Utilities for demo mode operation
"""

import hashlib
import numpy as np
from PIL import Image


def calculate_image_hash(image: Image.Image) -> str:
    """
    Calculate a unique hash for an image
    Used to identify demo images
    """
    # Resize to standard size for consistent hashing
    img_resized = image.resize((100, 100))
    img_array = np.array(img_resized)
    
    # Calculate hash
    img_bytes = img_array.tobytes()
    hash_obj = hashlib.md5(img_bytes)
    return hash_obj.hexdigest()


def save_demo_image_info(image_path: str):
    """
    Helper to get hash of a demo image
    Run this on your demo images to get their hashes
    """
    img = Image.open(image_path)
    img_hash = calculate_image_hash(img)
    print(f"Image: {image_path}")
    print(f"Hash: {img_hash}")
    print(f"Size: {img.size}")
    return img_hash
