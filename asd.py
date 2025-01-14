import cv2
import time
import math
import cvzone
from ultralytics import YOLO

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
confidence = 0.5  # Confidence threshold
classNames = ["fake", "real"]  # Class names

# Load YOLO model
model = YOLO(r"C:\Users\DELL\VISHNU_AM\Projects\ML-PY-models\Liveness-detection\models\l_version_1_300.pt")  # Replace with your trained YOLO model

prev_frame_time = 0

while True:
    # Capture frame from video
    success, img = cap.read()
    if not success:
        print("Error: Unable to read from the video source.")
        break

    new_frame_time = time.time()

    # Run YOLO model
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])

            # Display only "fake" objects
            if conf > confidence and classNames[cls] == 'fake':
                color = (0, 0, 255)  # Red for "fake"

                # Draw rectangle and put text
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(
                    img, f'{classNames[cls].lower()} {int(conf * 100)}%',
                    (max(0, x1), max(35, y1)), scale=0.5, thickness=1,
                    colorR=color, colorB=color
                )

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
    prev_frame_time = new_frame_time
    cv2.putText(
        img, f'FPS: {int(fps)}',
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
    )

    # Display the image
    cv2.imshow("Image", img)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
