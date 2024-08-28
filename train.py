from ultralytics import YOLO

model = YOLO('yolov5m6.pt')

model.train(epochs=50,data='data_custom.yaml',imgsz=640)
