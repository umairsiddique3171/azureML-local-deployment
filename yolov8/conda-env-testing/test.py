import json
import numpy as np
import torch
from ultralytics import YOLO
import cv2

def load_model(model_path):
    model = YOLO(model_path)
    return model

model = load_model('best3.pt')

def init():
    global model
    print("Model loaded successfully.")

def draw_bounding_boxes(image, predictions):
    for pred in predictions:
        box = pred['box']
        confidence = pred['confidence']
        cls = pred['class']
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        label = f"{cls} {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def run(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to read.")

        results = model(image)
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

        draw_bounding_boxes(image, output)
        
        output_image_path = 'output_image.jpg'
        cv2.imwrite(output_image_path, image)

        return json.dumps({'predictions': output, 'output_image': output_image_path})

    except Exception as e:
        return json.dumps({'error': str(e)})

if __name__ == "__main__":
    init()
    test_image_path = "test.jpg"
    print(run(test_image_path))
