from ultralytics import YOLO
from common import DATA_YAML

model = YOLO('yolov8n.pt')
model.train(data=DATA_YAML, epochs=50, imgsz=256, batch=16, name='circle_square_exp')
