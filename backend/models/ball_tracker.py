from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional

class BallTracker:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize the ball tracker with YOLOv8 model.
        
        Args:
            model_path (str): Path to YOLOv8 model weights
        """
        self.model = YOLO(model_path)
        self.ball_class_id = 32  # COCO dataset class ID for sports ball
        
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the ball in a single frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Optional[Tuple[int, int, int, int]]: Bounding box (x1, y1, x2, y2) if ball detected, None otherwise
        """
        results = self.model(frame, classes=[self.ball_class_id], conf=0.3)
        
        if len(results[0].boxes) > 0:
            # Get the detection with highest confidence
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return (x1, y1, x2, y2)
        return None
    
    def track_ball(self, frame: np.ndarray, prev_ball_pos: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Track the ball using both detection and previous position.
        
        Args:
            frame (np.ndarray): Input frame
            prev_ball_pos (Optional[Tuple[int, int, int, int]]): Previous ball position
            
        Returns:
            Optional[Tuple[int, int, int, int]]: Current ball position
        """
        # Try to detect the ball
        ball_pos = self.detect_ball(frame)
        
        if ball_pos is None and prev_ball_pos is not None:
            # If detection fails but we have previous position, use motion prediction
            # Simple implementation - can be improved with Kalman filter
            return prev_ball_pos
            
        return ball_pos
    
    def is_ball_in_play(self, ball_pos: Optional[Tuple[int, int, int, int]], 
                       pitch_boundaries: Tuple[int, int, int, int]) -> bool:
        """
        Determine if the ball is in play based on its position relative to pitch boundaries.
        
        Args:
            ball_pos (Optional[Tuple[int, int, int, int]]): Ball position
            pitch_boundaries (Tuple[int, int, int, int]): Pitch boundaries (x1, y1, x2, y2)
            
        Returns:
            bool: True if ball is in play, False otherwise
        """
        if ball_pos is None:
            return False
            
        x1, y1, x2, y2 = ball_pos
        px1, py1, px2, py2 = pitch_boundaries
        
        # Check if ball center is within pitch boundaries
        ball_center_x = (x1 + x2) // 2
        ball_center_y = (y1 + y2) // 2
        
        return (px1 <= ball_center_x <= px2 and 
                py1 <= ball_center_y <= py2)



