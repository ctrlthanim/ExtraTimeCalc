import cv2
import numpy as np
import argparse
from typing import List, Dict, Optional, Tuple
import librosa
import os
import subprocess
import time
from ball_tracker import BallTracker
from pitch_mapper import PitchMapper
from whistle_detector import WhistleDetector
from utils import get_video_properties, frame_to_timestamp, calculate_ball_in_play_time, save_results

# --- Additions for audio playback ---
import sounddevice as sd
import soundfile as sf
import threading

def extract_audio_if_needed(video_path: str) -> str:
    """
    Extract audio from video using ffmpeg if .wav file does not exist.
    Returns the path to the .wav file.
    """
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

def draw_tracking_info(frame: np.ndarray, 
                      ball_pos: Optional[Tuple[int, int, int, int]],
                      pitch_boundaries: Tuple[int, int, int, int],
                      is_in_play: bool,
                      current_time: float,
                      reason: str = None,
                      speed_multiplier: float = 1.0) -> Tuple[np.ndarray, Dict[str, Tuple[int, int, int, int]]]:
    """
    Draw tracking information on the frame.
    """
    # Draw pitch boundaries
    px1, py1, px2, py2 = pitch_boundaries
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
    
    # Draw ball position if detected
    if ball_pos is not None:
        x1, y1, x2, y2 = ball_pos
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (0, 0, 255), -1)
    
    # Draw status
    status = "IN PLAY" if is_in_play else "OUT OF PLAY"
    status_color = (0, 255, 0) if is_in_play else (0, 0, 255)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Draw timestamp
    timestamp = frame_to_timestamp(int(current_time * 30), 30)  # Assuming 30 fps
    cv2.putText(frame, timestamp, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw reason if out of play
    if not is_in_play and reason:
        cv2.putText(frame, reason, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw speed control buttons
    button_height = 40
    button_width = 60
    button_spacing = 10
    start_x = 10
    start_y = frame.shape[0] - button_height - 10
    
    # Create buttons
    buttons = {}
    frame, buttons['speed_down'] = create_button(frame, "-", 
                                               (start_x, start_y), 
                                               (button_width, button_height),
                                               (50, 50, 50), (70, 70, 70))
    
    frame, buttons['speed_display'] = create_button(frame, f"{speed_multiplier}x", 
                                                  (start_x + button_width + button_spacing, start_y),
                                                  (button_width, button_height),
                                                  (30, 30, 30), (50, 50, 50))
    
    frame, buttons['speed_up'] = create_button(frame, "+", 
                                             (start_x + 2 * (button_width + button_spacing), start_y),
                                             (button_width, button_height),
                                             (50, 50, 50), (70, 70, 70))
    
    return frame, buttons

def determine_stoppage_reason(ball_pos: Optional[Tuple[int, int, int, int]], 
                            pitch_boundaries: Tuple[int, int, int, int],
                            whistle_detected: bool) -> str:
    """
    Determine the reason for a stoppage based on ball position and whistle detection.
    Only track fouls (whistle stops) and ignore throw-ins.
    """
    if whistle_detected:
        return "Foul/Whistle"
    
    # If no whistle and ball is out, it's a throw-in - we'll ignore these
    if ball_pos is None:
        return "Ignored (Ball not detected)"
        
    x1, y1, x2, y2 = ball_pos
    px1, py1, px2, py2 = pitch_boundaries
    
    # Check if ball is out of bounds
    ball_center_x = (x1 + x2) // 2
    ball_center_y = (y1 + y2) // 2
    
    if ball_center_x < px1 or ball_center_x > px2 or ball_center_y < py1 or ball_center_y > py2:
        return "Ignored (Throw-in)"
    
    return "Unknown stoppage"

def create_button(img: np.ndarray, text: str, pos: Tuple[int, int], size: Tuple[int, int], 
                 color: Tuple[int, int, int], hover_color: Tuple[int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Create a button on the image.
    """
    x, y = pos
    w, h = size
    rect = (x, y, w, h)
    
    # Draw button background
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2)
    
    return img, rect

def is_point_in_rect(point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
    """
    Check if a point is inside a rectangle.
    """
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

def play_audio(audio_path, start_time, speed_multiplier, stop_event):
    """
    Play audio from a given start_time at the given speed_multiplier. Stops when stop_event is set.
    """
    data, samplerate = sf.read(audio_path, dtype='float32')
    start_sample = int(start_time * samplerate)
    data = data[start_sample:]
    if speed_multiplier != 1.0:
        import librosa
        data = librosa.resample(data.T, orig_sr=samplerate, target_sr=int(samplerate * speed_multiplier)).T
        samplerate = int(samplerate * speed_multiplier)
    def callback(outdata, frames, time, status):
        if status:
            print(status)
        chunk = data[callback.idx:callback.idx+frames]
        if len(chunk) < frames:
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = 0
            raise sd.CallbackStop()
        else:
            outdata[:] = chunk
        callback.idx += frames
        if stop_event.is_set():
            raise sd.CallbackStop()
    callback.idx = 0
    with sd.OutputStream(samplerate=samplerate, channels=data.shape[1] if len(data.shape) > 1 else 1, callback=callback):
        while not stop_event.is_set() and callback.idx < len(data):
            time.sleep(0.1)

def process_video(video_path: str, output_path: str, speed_multiplier: float = 1.0):
    """
    Process a football match video to track all out-of-play periods (throw-ins, fouls, etc.).
    """
    # Initialize components
    ball_tracker = BallTracker()
    pitch_mapper = PitchMapper()
    whistle_detector = WhistleDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width, height, total_frames, fps = get_video_properties(video_path)
    
    # Calculate frame delay based on speed multiplier
    frame_delay = int(1000 / (fps * speed_multiplier))  # Delay in milliseconds
    
    # Extract audio (auto-extract if needed)
    audio_path = extract_audio_if_needed(video_path)
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    # Initialize tracking variables
    periods = []
    out_of_play_active = False
    out_of_play_start_time = None
    out_of_play_reason = None
    prev_ball_pos = None
    pitch_boundaries = None
    
    # Create window for visualization
    cv2.namedWindow('Football Time Tracker', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Football Time Tracker', 1280, 720)
    
    # --- Audio playback management ---
    stop_audio_event = threading.Event()
    audio_thread = None
    audio_lock = threading.Lock()
    audio_playback_time = [0.0]  # Use a list for mutability in closure

    def start_audio_thread(current_time, speed):
        nonlocal audio_thread, stop_audio_event
        # Stop previous audio thread if running
        if audio_thread is not None and audio_thread.is_alive():
            stop_audio_event.set()
            audio_thread.join()
        stop_audio_event = threading.Event()
        audio_thread = threading.Thread(target=play_audio, args=(audio_path, current_time, speed, stop_audio_event))
        audio_thread.start()

    # Start audio at the beginning
    start_audio_thread(0, speed_multiplier)

    # Set mouse callback
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            buttons = param.get('buttons', {})
            if is_point_in_rect((x, y), buttons.get('speed_up', (0, 0, 0, 0))):
                param['speed_multiplier'] = min(10.0, param['speed_multiplier'] + 1.0)
                param['frame_delay'] = int(1000 / (fps * param['speed_multiplier']))
                print(f"\nSpeed increased to {param['speed_multiplier']}x")
                # Restart audio from current time at new speed
                with param['audio_lock']:
                    param['restart_audio'](param['audio_playback_time'][0], param['speed_multiplier'])
            elif is_point_in_rect((x, y), buttons.get('speed_down', (0, 0, 0, 0))):
                param['speed_multiplier'] = max(1.0, param['speed_multiplier'] - 1.0)
                param['frame_delay'] = int(1000 / (fps * param['speed_multiplier']))
                print(f"\nSpeed decreased to {param['speed_multiplier']}x")
                # Restart audio from current time at new speed
                with param['audio_lock']:
                    param['restart_audio'](param['audio_playback_time'][0], param['speed_multiplier'])
    
    # Create callback parameters
    callback_params = {
        'speed_multiplier': speed_multiplier,
        'frame_delay': frame_delay,
        'buttons': {},
        'audio_playback_time': audio_playback_time,
        'audio_lock': audio_lock,
        'restart_audio': start_audio_thread
    }
    cv2.setMouseCallback('Football Time Tracker', mouse_callback, callback_params)
    
    # Process each frame
    frame_count = 0
    last_frame_time = time.time()
    
    while True:
        # Calculate how many frames to skip based on speed
        current_time = time.time()
        elapsed = current_time - last_frame_time
        frames_to_skip = int(elapsed * fps * callback_params['speed_multiplier'])
        
        if frames_to_skip > 0:
            # Skip frames to maintain speed
            for _ in range(frames_to_skip - 1):
                cap.grab()
                frame_count += 1
            
            # Read the frame we want to display
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            last_frame_time = current_time
            
            # Get current timestamp
            current_time = frame_count / fps
            
            # Detect pitch boundaries (do this periodically or when needed)
            if frame_count % 30 == 0 or pitch_boundaries is None:
                pitch_boundaries = pitch_mapper.detect_pitch_boundaries(frame)
                pitch_boundaries = pitch_mapper.refine_boundaries(frame, pitch_boundaries)
            
            # Track ball
            ball_pos = ball_tracker.track_ball(frame, prev_ball_pos)
            prev_ball_pos = ball_pos
            
            # Check if ball is in play
            is_in_play = ball_tracker.is_ball_in_play(ball_pos, pitch_boundaries)
            
            # Check for whistle in current time window
            window_start = max(0, int((current_time - 0.5) * sr))
            window_end = min(len(audio_data), int((current_time + 0.5) * sr))
            audio_window = audio_data[window_start:window_end]
            whistle_detected = whistle_detector.is_whistle(audio_window)
            
            # Get current reason for stoppage if out of play
            current_reason = determine_stoppage_reason(ball_pos, pitch_boundaries, whistle_detected)
            
            # Draw tracking information and get button positions
            frame, buttons = draw_tracking_info(frame, ball_pos, pitch_boundaries, is_in_play, 
                                              current_time, current_reason, 
                                              callback_params['speed_multiplier'])
            callback_params['buttons'] = buttons
            
            # Show frame
            cv2.imshow('Football Time Tracker', frame)
            
            # --- Out-of-play period logic ---
            if not out_of_play_active and (not is_in_play or whistle_detected):
                # Start out-of-play period
                out_of_play_active = True
                out_of_play_start_time = current_time
                out_of_play_reason = current_reason
            elif out_of_play_active and is_in_play:
                # End out-of-play period when ball is back in play
                periods.append({
                    'type': 'out_of_play',
                    'start_time': out_of_play_start_time,
                    'end_time': current_time,
                    'reason': out_of_play_reason
                })
                out_of_play_active = False
                out_of_play_start_time = None
                out_of_play_reason = None
            # (If out_of_play_active and not is_in_play, keep waiting)
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # Update audio playback time for sync
            with audio_lock:
                audio_playback_time[0] = current_time
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Close video and windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Handle last out-of-play period if still active
    if out_of_play_active and out_of_play_start_time is not None:
        periods.append({
            'type': 'out_of_play',
            'start_time': out_of_play_start_time,
            'end_time': frame_count / fps,
            'reason': out_of_play_reason
        })
    
    # Calculate results
    total_out_of_play_time = sum(period['end_time'] - period['start_time'] for period in periods if period['type'] == 'out_of_play')
    
    # Save results
    results = {
        'total_out_of_play_time': total_out_of_play_time,
        'periods': periods,
        'fps': fps
    }
    save_results(results, output_path)
    
    print(f"\nAnalysis complete!")
    print(f"Total out-of-play time: {total_out_of_play_time:.2f} seconds")
    print(f"Results saved to: {output_path}")
    
    # Stop audio playback
    if audio_thread is not None and audio_thread.is_alive():
        stop_audio_event.set()
        audio_thread.join()

def main():
    parser = argparse.ArgumentParser(description='Football Time Tracker')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', default='results.txt', help='Path to save results')
    parser.add_argument('--speed', type=float, default=1.0, help='Video playback speed multiplier (e.g., 3.0 for 3x speed)')
    args = parser.parse_args()
    
    process_video(args.video, args.output, args.speed)

if __name__ == '__main__':
    main()


