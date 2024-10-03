import os 
import json
import logging
from PIL import Image
import numpy as np
from ultralytics import YOLO
from io import BytesIO
import base64

def load_model(model_path):
    model = YOLO(model_path)
    return model

def init():
    global model
    logging.info("Script Initialized")
    model_path = "best.pt"
    model = load_model(model_path)
    print("Model loaded successfully.")
    logging.info("Initialization Complete")

def run(raw_data):
    try:
        logging.info("Request received")
        input_data = json.loads(raw_data)
        img_data = base64.b64decode(input_data['data'])
        image = Image.open(BytesIO(img_data)).convert("RGB")
        image_np = np.array(image)
        results = model(image_np)
        predictions = results[0].boxes.data.numpy()
        labels = results[0].names
        output = []
        for pred in predictions:
            box = pred[:4].tolist()
            conf = pred[4].item()
            cls = int(pred[5].item())
            output.append({
                'box': box,
                'confidence': conf,
                'class': labels[cls]
            })
        logging.info("Request processed")
        return json.dumps({'predictions': output})

    except Exception as e:
        return json.dumps({'error': str(e)})

