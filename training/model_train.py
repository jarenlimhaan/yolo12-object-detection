from ultralytics import YOLO

model = YOLO('yolo12n.pt')  # Load a custom YOLO model configuration

results = model.train(data='data_config.yaml', epochs=100 , imgsz=640 , workers=1, batch=3 )