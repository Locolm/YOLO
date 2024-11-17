from ultralytics import YOLO
model = YOLO("yolo-Weights/yolov8n.pt")
model.train(data="dataHelmet.yaml", epochs=50, batch=16, imgsz=640)