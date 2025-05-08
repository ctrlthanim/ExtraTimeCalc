import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict

def get_video_properties(video_path: str) -> Tuple[int, int, int, float]:
    """
    Get video properties including width, height, total frames, and FPS.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        Tuple[int, int, int, float]: Width, height, total frames, and FPS
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, total_frames, fps

def frame_to_timestamp(frame_number: int, fps: float) -> str:
    """
    Convert frame number to timestamp string.
    
    Args:
        frame_number (int): Frame number
        fps (float): Frames per second
        
    Returns:
        str: Timestamp in HH:MM:SS format
    """
    seconds = frame_number / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def calculate_ball_in_play_time(periods: List[Dict[str, float]]) -> float:
    """
    Calculate total ball-in-play time from periods.
    
    Args:
        periods (List[Dict[str, float]]): List of periods with start and end times
        
    Returns:
        float: Total ball-in-play time in seconds
    """
    total_time = 0.0
    for period in periods:
        if period.get('type') == 'in_play':
            total_time += period['end_time'] - period['start_time']
    return total_time

def save_results(results: Dict, output_path: str):
    """
    Save analysis results to a file.
    
    Args:
        results (Dict): Analysis results
        output_path (str): Path to save results
    """
    with open(output_path, 'w') as f:
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if 'total_out_of_play_time' in results:
            f.write(f"Total out-of-play time: {results['total_out_of_play_time']:.2f} seconds\n")
        elif 'total_foul_time' in results:
            f.write(f"Total foul time: {results['total_foul_time']:.2f} seconds\n")
        else:
            f.write(f"Total ball-in-play time: {results['total_ball_in_play_time']:.2f} seconds\n")
        f.write("\nPeriods:\n")
        for period in results['periods']:
            period_type = period['type']
            start_time = period['start_time']
            end_time = period['end_time']
            duration = end_time - start_time
            reason = period.get('reason', 'Unknown')
            
            # Format the output with timestamp and duration
            start_timestamp = frame_to_timestamp(int(start_time * results['fps']), results['fps'])
            end_timestamp = frame_to_timestamp(int(end_time * results['fps']), results['fps'])
            
            f.write(f"{period_type.upper()}:\n")
            f.write(f"  Time: {start_timestamp} - {end_timestamp}\n")
            f.write(f"  Duration: {duration:.2f} seconds\n")
            if period_type in ('foul', 'out_of_play', 'out_of_play_time'):
                f.write(f"  Reason: {reason}\n")
            f.write("\n")



