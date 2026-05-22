"""
Advanced Validation Module
Provides intelligent filtering and verification for object detections
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image


class DetectionValidator:
    """Advanced validation for object detections"""
    
    # Expected object sizes (in pixels, approximate)
    OBJECT_SIZES = {
        'cell phone': (100, 300),  # min, max dimension
        'cup': (80, 250),
        'bottle': (100, 400),
        'laptop': (300, 600),
        'keyboard': (300, 500),
        'mouse': (50, 150),
        'book': (150, 400),
        'car': (400, 2000),
        'person': (200, 2000),
    }
    
    # Expected color ranges (HSV)
    OBJECT_COLORS = {
        'cell phone': [(0, 0, 0), (180, 50, 255)],  # Dark/metallic colors
        'orange': [(5, 100, 100), (15, 255, 255)],  # Orange fruit
        'banana': [(20, 100, 100), (30, 255, 255)],  # Yellow
    }
    
    # Impossible indoor objects
    OUTDOOR_OBJECTS = {'car', 'truck', 'bus', 'airplane', 'train', 'boat', 'motorcycle', 'bicycle'}
    
    def __init__(self):
        pass
    
    def validate_size(self, detection: Dict, image_shape: Tuple) -> bool:
        """Validate object size is realistic"""
        class_name = detection['class']
        bbox = detection['bbox']
        
        if class_name not in self.OBJECT_SIZES:
            return True  # No size constraint
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        max_dim = max(width, height)
        
        min_size, max_size = self.OBJECT_SIZES[class_name]
        
        # Check if size is reasonable
        if max_dim < min_size * 0.5 or max_dim > max_size * 2:
            return False
        
        return True
    
    def validate_color(self, image: np.ndarray, detection: Dict) -> bool:
        """Validate object color matches expected range"""
        class_name = detection['class']
        
        if class_name not in self.OBJECT_COLORS:
            return True  # No color constraint
        
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Extract object region
        obj_region = image[y1:y2, x1:x2]
        
        if obj_region.size == 0:
            return True
        
        # Convert to HSV
        hsv = cv2.cvtColor(obj_region, cv2.COLOR_RGB2HSV)
        
        # Check if dominant color matches expected range
        lower, upper = self.OBJECT_COLORS[class_name]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # If >30% of pixels match expected color, it's valid
        match_ratio = np.sum(mask > 0) / mask.size
        return match_ratio > 0.3
    
    def validate_edges(self, image: np.ndarray, detection: Dict) -> bool:
        """Validate object has expected edge patterns"""
        class_name = detection['class']
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Extract object region
        obj_region = image[y1:y2, x1:x2]
        
        if obj_region.size == 0:
            return True
        
        # Convert to grayscale
        gray = cv2.cvtColor(obj_region, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Objects should have reasonable edge density (not too sparse, not too dense)
        if class_name in ['cell phone', 'laptop', 'book']:
            # Rectangular objects should have clear edges
            return 0.05 < edge_density < 0.4
        elif class_name in ['cup', 'bottle']:
            # Rounded objects have different edge patterns
            return 0.03 < edge_density < 0.3
        
        return True
    
    def validate_context(self, detections: List[Dict]) -> List[Dict]:
        """Apply context-aware filtering"""
        if not detections:
            return detections
        
        filtered = []
        
        # Check for person + handheld object combinations
        person_boxes = [d for d in detections if d['class'] == 'person']
        
        for det in detections:
            class_name = det['class']
            
            # Filter out unlikely combinations
            if class_name in self.OUTDOOR_OBJECTS:
                # Check if there are indoor objects nearby
                indoor_objects = [d for d in detections if d['class'] in ['couch', 'bed', 'dining table']]
                if indoor_objects:
                    # Likely a false positive (car detected indoors)
                    continue
            
            # Validate handheld objects are near person's hands
            if class_name in ['cell phone', 'cup', 'bottle', 'book']:
                if person_boxes:
                    # Check if object is within person's bounding box
                    near_person = False
                    for person in person_boxes:
                        if self._boxes_overlap(det['bbox'], person['bbox']):
                            near_person = True
                            break
                    
                    # If person detected but object not near them, might be wrong
                    if not near_person and det['confidence'] < 0.75:
                        continue
            
            filtered.append(det)
        
        return filtered
    
    def _boxes_overlap(self, box1: List[int], box2: List[int]) -> bool:
        """Check if two bounding boxes overlap"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Check for overlap
        if x1_max < x2_min or x2_max < x1_min:
            return False
        if y1_max < y2_min or y2_max < y1_min:
            return False
        
        return True
