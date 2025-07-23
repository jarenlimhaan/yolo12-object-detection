import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("training/runs/detect/train3/weights/best.pt")

# Choose source (0 for webcam, or path to video file)
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or replace with "path/to/video.mp4"

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(source=frame, conf=0.7, save=False, stream=True)

    # Visualize the results
    for r in results:
        annotated_frame = r.plot()  # Plot boxes on the frame

    # Display the frame
    cv2.imshow("YOLOv12 Detection", annotated_frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
