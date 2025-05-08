import argparse
import os
from backend.core.processor import VideoProcessor
from frontend.ui.player import VideoPlayer

def main():
    parser = argparse.ArgumentParser(description='Football Time Tracker')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', required=True, help='Path to output file')
    parser.add_argument('--speed', type=float, default=1.0, help='Initial playback speed')
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Extract audio if needed
    audio_path = processor.extract_audio(args.video)
    
    # Initialize player
    player = VideoPlayer(args.video, audio_path)
    player.speed_multiplier = args.speed
    
    # Start playback
    player.play(processor)

if __name__ == '__main__':
    main()


