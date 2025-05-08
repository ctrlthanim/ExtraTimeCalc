import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
import subprocess
from ..models.ball_tracker import BallTracker
from ..models.pitch_mapper import PitchMapper
from ..models.whistle_detector import WhistleDetector
from ..utils import get_video_properties, frame_to_timestamp

class VideoProcessor:
    def __init__(self):
        self.ball_tracker = BallTracker()
        self.pitch_mapper = PitchMapper()
        self.whistle_detector = WhistleDetector()
        
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg if .wav file does not exist."""
        base, _ = os.path.splitext(video_path)
        audio_path = base + '.wav'
        if not os.path.exists(audio_path):
            print(f"Extracting audio to {audio_path}...")
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                print(f"Failed to extract audio: {e}")
                raise RuntimeError("Audio extraction failed. Please ensure ffmpeg is installed and in your PATH.")
        return audio_path

    def process_frame(self, frame: np.ndarray, current_time: float) -> Dict:
        """Process a single frame and return detection results."""
        # Track ball
        ball_pos = self.ball_tracker.detect_ball(frame)
        
        # Get pitch boundaries
        pitch_boundaries = self.pitch_mapper.get_pitch_boundaries(frame)
        
        # Check for whistle
        whistle_detected = self.whistle_detector.detect_whistle(current_time)
        
        # Determine if ball is in play
        is_in_play = self._is_ball_in_play(ball_pos, pitch_boundaries)
        
        # Determine stoppage reason
        reason = self._determine_stoppage_reason(ball_pos, pitch_boundaries, whistle_detected)
        
        return {
            'ball_pos': ball_pos,
            'pitch_boundaries': pitch_boundaries,
            'is_in_play': is_in_play,
            'reason': reason,
            'whistle_detected': whistle_detected
        }
    
    def _is_ball_in_play(self, ball_pos: Optional[Tuple[int, int, int, int]], 
                        pitch_boundaries: Tuple[int, int, int, int]) -> bool:
        """Determine if the ball is in play based on its position and pitch boundaries."""
        if ball_pos is None:
            return False
            
        x1, y1, x2, y2 = ball_pos
        px1, py1, px2, py2 = pitch_boundaries
        
        ball_center_x = (x1 + x2) // 2
        ball_center_y = (y1 + y2) // 2
        
        return (px1 <= ball_center_x <= px2 and 
                py1 <= ball_center_y <= py2)
    
    def _determine_stoppage_reason(self, ball_pos: Optional[Tuple[int, int, int, int]], 
                                 pitch_boundaries: Tuple[int, int, int, int],
                                 whistle_detected: bool) -> str:
        """Determine the reason for a stoppage based on ball position and whistle detection."""
        if whistle_detected:
            return "Foul/Whistle"
        
        if ball_pos is None:
            return "Ignored (Ball not detected)"
            
        x1, y1, x2, y2 = ball_pos
        px1, py1, px2, py2 = pitch_boundaries
        
        ball_center_x = (x1 + x2) // 2
        ball_center_y = (y1 + y2) // 2
        
        if ball_center_x < px1 or ball_center_x > px2 or ball_center_y < py1 or ball_center_y > py2:
            return "Ignored (Throw-in)"
        
        return "Unknown stoppage" 