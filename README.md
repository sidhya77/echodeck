# EchoDeck 🎧

EchoDeck is a real-time, AI-driven dual-hand DJ controller that interprets ASL letter gestures into MP3 playback commands.

## Features
- Real-time gesture tracking via webcam using MediaPipe
- ASL letter classification via pretrained CNN
- Track control mapped to left and right hands
- MP3 playback and OpenCV-based visual feedback

## Setup
```bash
git clone https://github.com/yourusername/echodeck.git
cd echodeck
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python echodeck_main.py
```
