import argparse
import os
import cv2
import ultralytics
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
    parser = argparse.ArgumentParser(description="SmartTraffic AI - Vehicle detection and tracking using YOLOv8")
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
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "strongsort"],
        help="Object tracker to use with YOLOv8",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Resize video width for faster processing (keep frame shape)",
    )
    return parser.parse_args()


def draw_box(frame, box, label, track_id, confidence):
    x1, y1, x2, y2 = map(int, box)
    color = COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    id_text = f"ID:{track_id}" if track_id is not None else "ID:?"
    text = f"{label} {id_text} {confidence:.2f}"

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


def get_tracker_config(tracker_name):
    tracker_name = tracker_name.lower()
    valid_names = {"bytetrack", "botsort"}
    if tracker_name not in valid_names:
        raise ValueError(f"Tracker must be one of {valid_names}, got '{tracker_name}'")

    base_dir = os.path.dirname(ultralytics.__file__)
    cfg_path = os.path.join(base_dir, "cfg", "trackers", f"{tracker_name}.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Tracker configuration not found: {cfg_path}")
    return cfg_path


def main():
    args = parse_args()

    print("Loading YOLOv8 nano model...")
    model = YOLO("yolov8n.pt")

    print(f"Opening video source: {args.source}")
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    tracker_config = get_tracker_config(args.tracker)
    seen_ids = {}
    unique_counts = {label: 0 for label in VEHICLE_CLASSES.values()}

    while True:
        success, frame = capture.read()
        if not success:
            print("End of video or cannot read frame.")
            break

        if frame.shape[1] != args.width:
            scale = args.width / frame.shape[1]
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (args.width, new_height), interpolation=cv2.INTER_AREA)

        results = model.track(
            frame,
            tracker=tracker_config,
            conf=args.conf,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )
        result = results[0]

        frame_counts = {label: 0 for label in VEHICLE_CLASSES.values()}

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                label = VEHICLE_CLASSES[cls_id]
                confidence = float(box.conf[0])

                track_id = None
                if hasattr(box, "id") and box.id is not None:
                    try:
                        track_id = int(box.id[0])
                    except Exception:
                        track_id = int(box.id)

                if track_id is not None and track_id not in seen_ids:
                    seen_ids[track_id] = label
                    unique_counts[label] += 1

                if hasattr(box, "xyxy"):
                    coords = box.xyxy[0].cpu().numpy()
                else:
                    coords = box.xyxy

                draw_box(frame, coords, label, track_id, confidence)
                frame_counts[label] += 1

        total_unique = sum(unique_counts.values())

        cv2.putText(
            frame,
            f"Cars: {frame_counts['Car']}  Buses: {frame_counts['Bus']}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Trucks: {frame_counts['Truck']}  Bikes: {frame_counts['Bike']}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Frame total: {sum(frame_counts.values())}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Unique Cars: {unique_counts['Car']}  Unique Buses: {unique_counts['Bus']}",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Unique Trucks: {unique_counts['Truck']}  Unique Bikes: {unique_counts['Bike']}",
            (15, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Total Vehicles: {total_unique}",
            (15, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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
