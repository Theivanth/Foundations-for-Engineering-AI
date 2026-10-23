import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


OFFICE_SUPPLIES_CLASS_NAMES = {
    0: "tape dispenser",
    1: "highlighter",
    2: "glue stick",
    3: "pen",
    4: "scissors",
    5: "protractor",
    6: "correction tape",
}


def main():
    parser = argparse.ArgumentParser(
        description="MLE1003 Lesson 6 Video Object Detection"
    )

    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Path to trained YOLO model",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or video path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Minimum detection confidence",
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Download best.pt from Kaggle first."
        )

    model = YOLO(str(model_path))
    if set(model.names) != set(OFFICE_SUPPLIES_CLASS_NAMES):
        raise ValueError(
            "The office-supplies label mapping requires model classes 0 through 6."
        )

    # Numeric input uses a webcam.
    # Otherwise, use a video file path.
    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video source: {source}"
        )

    previous_time = time.perf_counter()
    frames_processed = 0

    print("Video recognition started.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Run detection on one frame.
            results = model.predict(
                frame,
                conf=args.conf,
                device="cpu",
                verbose=False,
            )

            result = results[0]
            result.names = OFFICE_SUPPLIES_CLASS_NAMES

            # Draw predicted bounding boxes.
            annotated = result.plot()

            now = time.perf_counter()
            elapsed = max(now - previous_time, 1e-6)
            previous_time = now

            fps = 1.0 / elapsed

            # Show processing speed.
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # Count detections.
            object_count = len(result.boxes)

            cv2.putText(
                annotated,
                f"Objects: {object_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "MLE1003 - Video Object Recognition",
                annotated,
            )

            frames_processed += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Total frames processed:", frames_processed)


if __name__ == "__main__":
    main()