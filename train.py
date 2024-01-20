from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model

model.train(data='config.yaml', epochs=1, imgsz=640)
