from ultralytics import YOLO
import cv2
import math
import tkinter as tk
from tkinter import filedialog, messagebox

'''This code is for running the custom trained YOLO model on the webcam feed.
it can detect objects on images, videos, and webcam feeds and display the class name and confidence score on the detections.
In addition we have the probability of the detected objects displayed on the terminal.
You can select the detection mode (webcam, video, or image) using a GUI dialog box.
There is a video and an image exemple on yolo_img_video_tests folder.'''

# Load your custom trained YOLO model
model = YOLO('runs/detect/train/weights/best.pt')  # Path to the best model from training

# Class names from your dataset
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']


def detect_on_image(image_path):
    """Run detection on an uploaded image."""
    img = cv2.imread(image_path)
    results = model(img)

    # Draw results on the image
    probabilities = {}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Display class name and probability
            cls = int(box.cls[0])
            conf = box.conf[0]
            label = f"{classNames[cls]} ({conf:.2f})"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Track probabilities
            if classNames[cls] not in probabilities:
                probabilities[classNames[cls]] = []
            probabilities[classNames[cls]].append(round(conf.item(), 2))

    # Print probabilities to the terminal
    print("\nProbability:")
    for cls, confs in probabilities.items():
        print(f"- {cls}: {confs}")

    # Display the image with detections
    cv2.imshow("Image Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_frame(frame, max_width=800, max_height=600):
    """Resize a frame to fit within max dimensions."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def detect_on_video(video_path, process_every_n_frames=15):
    """Run detection on an uploaded video, processing every n-th frame, with persistent bounding boxes."""
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0  # Initialize frame counter
    persistent_boxes = []  # Store bounding boxes and labels
    persistent_probs = {}  # Store probabilities for display

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1

        # Update YOLO detections every n frames
        if frame_counter % process_every_n_frames == 0:
            persistent_boxes = []  # Clear old boxes
            persistent_probs = {}  # Clear probabilities

            # Run inference
            results = model(frame)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Extract bounding box and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = box.conf[0]

                    # Append box and label
                    persistent_boxes.append((x1, y1, x2, y2, classNames[cls], conf))

                    # Track probabilities for terminal output
                    if classNames[cls] not in persistent_probs:
                        persistent_probs[classNames[cls]] = []
                    persistent_probs[classNames[cls]].append(round(conf.item(), 2))

            # Print updated probabilities
            print("\nProbability (updated):")
            for cls, confs in persistent_probs.items():
                print(f"- {cls}: {confs}")

        # Draw persistent bounding boxes on the current frame
        for box in persistent_boxes:
            x1, y1, x2, y2, label, conf = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Resize the frame for display
        resized_frame = resize_frame(frame)

        # Show video frame with persistent detections
        cv2.imshow("Video Detection", resized_frame)

        if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('Video Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_on_webcam():
    """Run detection on a webcam feed."""
    cap = cv2.VideoCapture(0)  # 0 for the default webcam

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img)
        probabilities = {}

        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Display class name and probability
                cls = int(box.cls[0])
                conf = box.conf[0]
                label = f"{classNames[cls]} ({conf:.2f})"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Track probabilities
                if classNames[cls] not in probabilities:
                    probabilities[classNames[cls]] = []
                probabilities[classNames[cls]].append(round(conf.item(), 2))

        # Print probabilities to the terminal
        print("\nProbability:")
        for cls, confs in probabilities.items():
            print(f"- {cls}: {confs}")

        # Show webcam feed with detections
        cv2.imshow("Webcam Detection", img)

        if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('Webcam Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def select_mode():
    """Display a dialog box to select detection mode."""
    def on_webcam():
        root.destroy()  # Close the GUI
        detect_on_webcam()

    def on_video():
        root.destroy()  # Close the GUI
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if video_path:
            detect_on_video(video_path)
        else:
            messagebox.showerror("Error", "No video file selected.")

    def on_image():
        root.destroy()  # Close the GUI
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if image_path:
            detect_on_image(image_path)
        else:
            messagebox.showerror("Error", "No image file selected.")

    # Create a GUI window
    root = tk.Tk()
    root.title("Select Detection Mode")

    tk.Label(root, text="Choose an option for object detection:").pack(pady=10)

    tk.Button(root, text="Webcam", command=on_webcam, width=20).pack(pady=5)
    tk.Button(root, text="Video", command=on_video, width=20).pack(pady=5)
    tk.Button(root, text="Image", command=on_image, width=20).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    select_mode()
