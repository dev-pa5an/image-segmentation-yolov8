from ultralytics import YOLOV8n

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='config.yaml', epochs=1, imgsz=640)
