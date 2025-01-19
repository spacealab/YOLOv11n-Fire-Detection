from ultralytics import YOLO

# Load a YOLO model
model = YOLO("yolo11n.pt")

# Train the model
model.train(data="datasets/combined_data.yaml", epochs=10, imgsz=640, device="cpu")