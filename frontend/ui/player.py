import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
from typing import Dict, Tuple, Optional

class VideoPlayer:
    def __init__(self, video_path: str, audio_path: str):
        self.video_path = video_path
        self.audio_path = audio_path
        self.cap = cv2.VideoCapture(video_path)
        self.speed_multiplier = 1.0
        self.stop_event = threading.Event()
        self.audio_thread = None
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize window
        self.window_name = "Football Time Tracker"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Initialize buttons
        self.buttons = self._create_buttons()
    
    def _create_buttons(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Create control buttons for the player."""
        button_height = 40
        button_width = 60
        button_spacing = 10
        start_x = 10
        start_y = self.height - button_height - 10
        
        return {
            'speed_down': (start_x, start_y, button_width, button_height),
            'speed_display': (start_x + button_width + button_spacing, start_y, button_width, button_height),
            'speed_up': (start_x + 2 * (button_width + button_spacing), start_y, button_width, button_height)
        }
    
    def _draw_buttons(self, frame: np.ndarray) -> np.ndarray:
        """Draw control buttons on the frame."""
        for name, (x, y, w, h) in self.buttons.items():
            color = (50, 50, 50)
            if name == 'speed_display':
                text = f"{self.speed_multiplier}x"
            else:
                text = "-" if name == 'speed_down' else "+"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for button clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (bx, by, bw, bh) in self.buttons.items():
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    if name == 'speed_down':
                        self.speed_multiplier = max(1.0, self.speed_multiplier - 1.0)
                    elif name == 'speed_up':
                        self.speed_multiplier = min(10.0, self.speed_multiplier + 1.0)
                    self._restart_audio()
    
    def _restart_audio(self):
        """Restart audio playback with new speed."""
        if self.audio_thread and self.audio_thread.is_alive():
            self.stop_event.set()
            self.audio_thread.join()
        self.stop_event.clear()
        current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.audio_thread = threading.Thread(
            target=self._play_audio,
            args=(current_time, self.speed_multiplier)
        )
        self.audio_thread.start()
    
    def _play_audio(self, start_time: float, speed_multiplier: float):
        """Play audio from a given start time at the given speed."""
        data, samplerate = sf.read(self.audio_path, dtype='float32')
        start_sample = int(start_time * samplerate)
        data = data[start_sample:]
        
        if speed_multiplier != 1.0:
            import librosa
            data = librosa.resample(data.T, orig_sr=samplerate, 
                                  target_sr=int(samplerate * speed_multiplier)).T
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
            if self.stop_event.is_set():
                raise sd.CallbackStop()
        
        callback.idx = 0
        with sd.OutputStream(samplerate=samplerate, 
                           channels=data.shape[1] if len(data.shape) > 1 else 1,
                           callback=callback):
            while not self.stop_event.is_set() and callback.idx < len(data):
                time.sleep(0.1)
    
    def play(self, frame_processor):
        """Main playback loop."""
        current_time = 0
        self._restart_audio()
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            results = frame_processor.process_frame(frame, current_time)
            
            # Draw results
            frame = self._draw_results(frame, results)
            
            # Draw controls
            frame = self._draw_buttons(frame)
            
            # Show frame
            cv2.imshow(self.window_name, frame)
            
            # Handle key press
            key = cv2.waitKey(int(1000 / (self.fps * self.speed_multiplier)))
            if key == ord('q'):
                break
            
            current_time += 1 / self.fps
        
        self.cleanup()
    
    def _draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw processing results on the frame."""
        # Draw pitch boundaries
        px1, py1, px2, py2 = results['pitch_boundaries']
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        
        # Draw ball position if detected
        if results['ball_pos'] is not None:
            x1, y1, x2, y2 = results['ball_pos']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (0, 0, 255), -1)
        
        # Draw status
        status = "IN PLAY" if results['is_in_play'] else "OUT OF PLAY"
        status_color = (0, 255, 0) if results['is_in_play'] else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Draw timestamp
        timestamp = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
        cv2.putText(frame, timestamp, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw reason if out of play
        if not results['is_in_play'] and results['reason']:
            cv2.putText(frame, results['reason'], (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_event.set()
        if self.audio_thread:
            self.audio_thread.join()
        self.cap.release()
        cv2.destroyAllWindows() 