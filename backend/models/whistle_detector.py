import librosa
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple

class WhistleDetector:
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the whistle detector.
        
        Args:
            sample_rate (int): Audio sample rate
        """
        self.sample_rate = sample_rate
        # Whistle frequency range (typical football whistle is around 2-4 kHz)
        self.min_freq = 2000
        self.max_freq = 4000
        
    def detect_whistles(self, audio_data: np.ndarray) -> List[float]:
        """
        Detect whistle sounds in audio data.
        
        Args:
            audio_data (np.ndarray): Audio data array
            
        Returns:
            List[float]: List of timestamps where whistles were detected
        """
        # Compute spectrogram
        D = librosa.stft(audio_data)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Focus on whistle frequency range
        freq_bins = librosa.fft_frequencies(sr=self.sample_rate)
        mask = (freq_bins >= self.min_freq) & (freq_bins <= self.max_freq)
        S_db_filtered = S_db[mask]
        
        # Find peaks in the filtered spectrogram
        peaks, _ = find_peaks(np.mean(S_db_filtered, axis=0), 
                            height=-30,  # Adjust threshold as needed
                            distance=self.sample_rate // 2)  # Minimum distance between peaks
        
        # Convert peak indices to timestamps
        times = librosa.times_like(S_db_filtered)
        whistle_times = [times[i] for i in peaks]
        
        return whistle_times
    
    def is_whistle(self, audio_segment: np.ndarray) -> bool:
        """
        Check if a short audio segment contains a whistle.
        
        Args:
            audio_segment (np.ndarray): Short audio segment
            
        Returns:
            bool: True if whistle detected, False otherwise
        """
        # Compute spectrogram for the segment
        D = librosa.stft(audio_segment)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Focus on whistle frequency range
        freq_bins = librosa.fft_frequencies(sr=self.sample_rate)
        mask = (freq_bins >= self.min_freq) & (freq_bins <= self.max_freq)
        S_db_filtered = S_db[mask]
        
        # Check if there's significant energy in the whistle frequency range
        energy = np.mean(S_db_filtered)
        return energy > -30  # Adjust threshold as needed
    
    def get_whistle_segments(self, audio_data: np.ndarray, 
                           window_size: int = 2048) -> List[Tuple[float, float]]:
        """
        Get segments of audio containing whistles.
        
        Args:
            audio_data (np.ndarray): Audio data array
            window_size (int): Size of analysis window
            
        Returns:
            List[Tuple[float, float]]: List of (start_time, end_time) tuples for whistle segments
        """
        whistle_times = self.detect_whistles(audio_data)
        segments = []
        
        if not whistle_times:
            return segments
        
        # Group nearby whistles into segments
        current_segment = [whistle_times[0]]
        
        for time in whistle_times[1:]:
            if time - current_segment[-1] < 1.0:  # 1 second threshold
                current_segment.append(time)
            else:
                segments.append((current_segment[0], current_segment[-1]))
                current_segment = [time]
        
        if current_segment:
            segments.append((current_segment[0], current_segment[-1]))
        
        return segments



