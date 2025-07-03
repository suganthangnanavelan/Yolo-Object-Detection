from ultralytics import YOLO
import numpy as np
from PIL import Image

def detect_image(image: Image.Image, model_path="models/yolov8n.pt", conf=0.4):
    model = YOLO(model_path)
    image_np = np.array(image)
    results = model(image_np, conf=conf)
    return results[0].plot()
