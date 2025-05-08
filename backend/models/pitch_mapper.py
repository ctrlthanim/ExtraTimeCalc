import cv2
import numpy as np
from typing import Tuple, List

class PitchMapper:
    def __init__(self):
        """
        Initialize the pitch mapper with default parameters.
        """
        # HSV color ranges for grass detection
        self.lower_green = np.array([35, 50, 50])
        self.upper_green = np.array([85, 255, 255])
        
    def detect_pitch_boundaries(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect pitch boundaries using color segmentation.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Tuple[int, int, int, int]: Pitch boundaries (x1, y1, x2, y2)
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green areas
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, return full frame dimensions
            height, width = frame.shape[:2]
            return (0, 0, width, height)
        
        # Find the largest contour (assumed to be the pitch)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some padding to ensure we don't cut off the pitch
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        return (x1, y1, x2, y2)
    
    def refine_boundaries(self, frame: np.ndarray, initial_boundaries: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Refine pitch boundaries using additional processing.
        
        Args:
            frame (np.ndarray): Input frame
            initial_boundaries (Tuple[int, int, int, int]): Initial boundaries
            
        Returns:
            Tuple[int, int, int, int]: Refined boundaries
        """
        x1, y1, x2, y2 = initial_boundaries
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create mask for green areas
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        # Find edges
        edges = cv2.Canny(mask, 50, 150)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Use lines to refine boundaries
            # This is a simplified version - can be improved with more sophisticated line analysis
            for line in lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                # Update boundaries based on line positions
                x1 = min(x1, x1_line + x1)
                y1 = min(y1, y1_line + y1)
                x2 = max(x2, x2_line + x1)
                y2 = max(y2, y2_line + y1)
        
        return (x1, y1, x2, y2)



