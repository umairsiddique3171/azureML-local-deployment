import torch
from PIL import Image
import matplotlib.pyplot as plt

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

img_path = 'test.jpg'
img = Image.open(img_path)

results = model(img)

results.save()

