from ultralytics import YOLO
import cv2
import math

# Load your custom trained YOLO model
model = YOLO('runs/detect/train2/weights/best.pt')  # Path to the best model from training

# Start webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam, change if you have multiple cameras
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Class names from your dataset
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

while True:
    success, img = cap.read()  # Capture frame
    results = model(img)  # Run inference with YOLOv8

    # Process the results
    for r in results:
        boxes = r.boxes  # Get detected bounding boxes

        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integer values

            # Draw bounding box on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Get confidence score
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence:", confidence)

            # Get class name
            cls = int(box.cls[0])
            print("Class:", classNames[cls])

            # Display class name on the image
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Show the webcam feed with detections
    cv2.imshow('Webcam', img)

    # Check if 'q' is pressed or the window is closed
    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
