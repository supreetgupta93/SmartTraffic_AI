# SmartTraffic AI

A beginner-friendly AI-based traffic monitoring system for real-time vehicle detection and counting using YOLOv8 and OpenCV.

---

## Features

- Real-time vehicle detection
- Vehicle counting system
- YOLOv8 object detection
- OpenCV video processing
- Bounding box visualization
- Beginner-friendly implementation

---

## Tech Stack

- Python
- OpenCV
- YOLOv8
- Ultralytics

---

## Folder Structure

```bash
SmartTraffic_AI/
├── assets/
│   └── output.png        # Output screenshot
├── smart_traffic.py      # Main detection script
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Ignore unnecessary files
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/supreetgupta93/SmartTraffic_AI.git
cd SmartTraffic_AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

Place your traffic video locally and run:

```bash
python smart_traffic.py --source assets/traffic.mp4 --tracker bytetrack
```

If your video is large, you can also resize frames for faster processing:

```bash
python smart_traffic.py --source assets/traffic.mp4 --tracker bytetrack --width 960
```

Press `Esc` to close the OpenCV window.

---

## How YOLOv8 Works

YOLOv8 is a real-time object detection model that processes images in a single pass. It detects vehicles by predicting:

- Bounding boxes
- Object classes
- Confidence scores

The project uses `yolov8n.pt` (nano model) for faster performance and lightweight execution.

---

## How Vehicle Counting Works

The system detects vehicles such as:

- Cars
- Bikes
- Trucks
- Buses

For each frame:
1. YOLOv8 detects vehicles
2. Bounding boxes are drawn
3. Detected vehicles are counted
4. Total count is displayed on screen

---

## Input

- Traffic video
- CCTV footage
- Road camera feed

---

## Output

- Vehicle detection boxes
- Vehicle labels
- Total vehicle count
- Real-time traffic monitoring

---

## Future Improvements

- Traffic congestion analysis
- Speed detection
- Accident detection
- Smart traffic light control
- Multi-camera support

---

## Demo Video

Add your Google Drive or YouTube demo link here.

---

## Author

Supreet Gupta
