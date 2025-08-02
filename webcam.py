import cv2
from ultralytics import YOLO

# Load model
model = YOLO("runs/detect/train/weights/best.pt")

# Required object labels
required_labels = {"human", "vest", "helmet"}

# Capture from webcam
cap = cv2.VideoCapture(0)

# Load model class names (id: name)
class_names = model.model.names  # e.g., {0: 'human', 1: 'vest', 2: 'helmet'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict on current frame
    results = model.predict(source=frame, conf=0.7, save=False, stream=True)

    for r in results:
        # Draw detections on frame
        annotated_frame = r.plot()

        # Get detected class IDs
        if r.boxes is not None and r.boxes.cls is not None:
            detected_ids = r.boxes.cls.int().tolist()
            detected_labels = {class_names[i] for i in detected_ids}
        else:
            detected_labels = set()

        # Compare with required labels
        missing_labels = required_labels - detected_labels

        if missing_labels:
            # Red border
            h, w, _ = annotated_frame.shape
            cv2.rectangle(annotated_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

            # Display which labels are missing
            cv2.putText(
                annotated_frame,
                f"Missing: {', '.join(missing_labels)}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        # Optional: show detected labels for debug
        label_text = f"Detected: {', '.join(detected_labels) if detected_labels else 'None'}"
        cv2.putText(
            annotated_frame,
            label_text,
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # Show the result
    cv2.imshow("YOLO Detection", annotated_frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
