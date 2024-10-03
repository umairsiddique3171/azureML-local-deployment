import os
import json
import logging
from PIL import Image
import torch
from io import BytesIO
import base64

def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model

def init():
    global model
    logging.info("Script Initialized")
    model_path = "best.pt"
    model = load_model(model_path)
    logging.info("Model loaded successfully")
    logging.info("Initialization Complete")

def run(raw_data):
    try:
        logging.info("Request received")
        input_data = json.loads(raw_data)
        img_data = base64.b64decode(input_data['data'])
        image = Image.open(BytesIO(img_data)).convert("RGB")

        results = model(image)
        results_df = results.pandas().xywh[0]
        predictions = []

        for _, row in results_df.iterrows():
            box = [row['xcenter'], row['ycenter'], row['width'], row['height']]
            conf = row['confidence']
            cls = row['class']
            name = row['name']
            predictions.append({
                'box': box,
                'confidence': conf,
                'class': name
            })

        logging.info("Request processed")
        return json.dumps({'predictions': predictions})

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return json.dumps({'error': str(e)})
