# SmartTraffic AI

A beginner-friendly Python project for real-time vehicle detection and counting using YOLOv8 and OpenCV.

## Folder Structure

```
SmartTraffic_AI/
├── assets/
│   └── traffic.mp4        # Your sample traffic video file
├── smart_traffic.py      # Main detection script
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── .gitignore            # Ignore files for Git
```

## Installation

1. Install Python 3.10 or newer.
2. Open a terminal in the `SmartTraffic_AI` folder.
3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the environment:

- Windows:
  ```bash
  venv\Scripts\activate
  ```
- macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Place your traffic video in `assets/` and name it `traffic.mp4` or update the `--source` path.

## Run the project

```bash
python smart_traffic.py --source assets/traffic.mp4
```

Press `Esc` to stop the live OpenCV window.

## How YOLOv8 works

YOLOv8 is an object detection model that processes each image in a single pass. It divides the image into grid cells and predicts bounding boxes, class labels, and confidence scores simultaneously. The `yolov8n.pt` model is the nano variant, optimized for fast inference on CPU while still providing good detection accuracy.

## How vehicle counting works

The script detects vehicle classes such as cars, motorcycles, bicycles, buses, and trucks. For every frame, it draws bounding boxes and counts the detection results. The total number of vehicles in the current frame is displayed on the screen.

## Notes

- If you want to use your own video file, change the `--source` path when running the script.
- `yolov8n.pt` will be downloaded automatically by the `ultralytics` package if it is not already present.
"# SmartTraffic_AI" 
"# SmartTraffic_AI" 
