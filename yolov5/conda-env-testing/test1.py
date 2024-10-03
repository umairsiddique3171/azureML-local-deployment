import torch
from PIL import Image
import pandas 

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best1.pt')

img_path = 'test.jpg'
img = Image.open(img_path)

results = model(img)

results.save()

results_df = results.pandas().xywh[0]
print("Detected objects:")
result_df = results_df[['name', 'confidence', 'xcenter', 'ycenter', 'width', 'height', 'class']]
result_df.to_csv("predictions.csv")

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

print(predictions)

