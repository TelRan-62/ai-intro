from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='datasets/data.yaml', epochs=20, imgsz=256, batch=2, name='circle_exp')
