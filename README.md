# Football Time Tracker

This project is a football (soccer) video analysis tool that automatically detects and tracks all out-of-play periods in a match video—including throw-ins, fouls, and any time the ball is out of play. It calculates the total out-of-play time and provides a synchronized demo with both video and audio playback, featuring interactive speed controls.

## Features
- **Automatic detection of out-of-play periods** (throw-ins, fouls, etc.) using computer vision and audio analysis.
- **Ball tracking** with YOLOv8 and pitch boundary detection.
- **Whistle detection** for accurate foul and stoppage timing.
- **Synchronized video and audio playback** in the demo window.
- **Interactive speed controls**: Change playback speed (1x–10x) using on-screen buttons; audio stays in sync.
- **Detailed results**: Outputs a summary of all stoppages, their reasons, and total out-of-play time.

## Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd football-time-tracker
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install sounddevice soundfile
   ```
3. [Install FFmpeg](https://ffmpeg.org/download.html) and ensure it is in your system PATH (for audio extraction).

## Usage
1. Place your football match video (e.g., `match.mp4`) in the project directory.
2. Run the tracker:
   ```bash
   python main.py --video match.mp4 --output results.txt --speed 3
   ```
   - Use the on-screen `+` and `-` buttons to change playback speed during the demo. The audio will stay in sync.
   - Press `q` to quit the demo.
3. After analysis, check `results.txt` for a summary of all out-of-play periods and total out-of-play time.

## Output Example
```
Analysis completed at: 2024-06-10 15:00:00
Total out-of-play time: 123.45 seconds

Periods:
OUT_OF_PLAY:
  Time: 00:05:12 - 00:05:45
  Duration: 33.00 seconds
  Reason: Ball out of play (throw-in)

OUT_OF_PLAY:
  Time: 00:10:10 - 00:10:40
  Duration: 30.00 seconds
  Reason: Foul/Whistle
...
```

## Notes
- For best results, use a video clip with minimal replays and overlays.
- The tool is designed for raw match footage, not highlight reels.
- You can adjust whistle detection sensitivity in `whistle_detector.py` if needed.

## Requirements
- Python 3.8+
- OpenCV, numpy, librosa, sounddevice, soundfile, ffmpeg

## License
MIT 