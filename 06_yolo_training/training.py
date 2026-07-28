from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='datasets/data.yaml', epochs=50, imgsz=256, batch=16, name='circle_exp')
