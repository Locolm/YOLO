# YOLO Security Detection Project
This repository contains scripts to train a YOLO model for security detection and to run object detection on images, videos, or webcam streams. The trained model identifies whether individuals are wearing appropriate safety equipment (e.g., hardhats, safety vests).

## Features
- Real-time object detection: Detect safety gear in images, videos, or webcam feeds.
- Custom YOLO training: Train a YOLO model to learn specific security-related objects.
- Versatile input sources: Supports detection from webcam, video files, and images.

## File Descriptions
### captureVideo.py
*Purpose*: Runs the YOLO detection model on a webcam feed.

*Usage*: Detects various objects, including hardhats, safety vests, and more, in real-time through your webcam.

### train.py
*Purpose*: Trains a YOLO model using custom-labeled data for security detection.

*Usage*: Prepares a YOLO model to detect whether individuals are wearing essential safety gear. Adjusts to detect specific objects defined in your dataset.

*warning*: training the model can take some time, it took 9hours on CPU 12th Gen Intel(R) Core(TM) i9-12900H 2.90 GHz.

### runCustom.py
*Purpose*: Runs the trained YOLO model on different input sources (images, videos, or webcam).

*Usage*: Detects safety objects on individuals.
Supports input via image files, video files, or live webcam feeds.
Provides flexibility to process and analyze different types of media.

### datas
You can add label data to train your own model, in order to do so you need to modify *train.py* with your correct yaml :

model.train(data="dataHelmet.yaml", epochs=50, batch=16, imgsz=640)

and the *runCustom.py* file path :

model = YOLO('runs/detect/train/weights/best.pt')