import argparse
import cv2
from ultralytics import YOLO

# Map COCO class IDs to the vehicle label we want to show.
VEHICLE_CLASSES = {
    1: "Bike",       # bicycle
    2: "Car",        # car
    3: "Bike",       # motorcycle
    5: "Bus",        # bus
    7: "Truck",      # truck
}

# Colors for bounding boxes (BGR format).
COLORS = {
    "Car": (0, 255, 0),
    "Bus": (0, 165, 255),
    "Truck": (0, 0, 255),
    "Bike": (255, 0, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="SmartTraffic AI - Vehicle detection using YOLOv8")
    parser.add_argument(
        "--source",
        type=str,
        default="assets/traffic.mp4",
        help="Path to the traffic video file or camera index (0 for webcam)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for detections",
    )
    return parser.parse_args()


def draw_box(frame, box, label, confidence):
    x1, y1, x2, y2 = map(int, box)
    color = COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence:.2f}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    text_origin = (x1, y1 - 10 if y1 - 10 > 20 else y1 + 20)
    cv2.rectangle(
        frame,
        (text_origin[0], text_origin[1] - text_size[1] - 4),
        (text_origin[0] + text_size[0], text_origin[1] + 4),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (text_origin[0], text_origin[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    print("Loading YOLOv8 nano model...")
    model = YOLO("yolov8n.pt")

    print(f"Opening video source: {args.source}")
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    while True:
        success, frame = capture.read()
        if not success:
            print("End of video or cannot read frame.")
            break

        # Run YOLO inference on the current frame.
        results = model(frame, conf=args.conf, classes=list(VEHICLE_CLASSES.keys()))
        result = results[0]

        count = 0
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                label = VEHICLE_CLASSES[cls_id]
                confidence = float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy()
                draw_box(frame, coords, label, confidence)
                count += 1

        # Display the total vehicle count on the frame.
        cv2.putText(
            frame,
            f"Total Vehicles: {count}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("SmartTraffic AI - Vehicle Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key to exit
            print("Exit requested by user.")
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
