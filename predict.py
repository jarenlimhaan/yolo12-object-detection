from ultralytics import YOLO

model = YOLO("training/runs/detect/train3/weights/best.pt")

import os 

# Loop through the images in the 'images' directory
image_dir = 'predictions'
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        results = model.predict(image_path, save=True, show=True, conf=0.7, save_txt=True)
