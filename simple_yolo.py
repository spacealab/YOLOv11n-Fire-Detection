from ultralytics import YOLO

# Load a YOLO model (using YOLOv11s for better performance)
model = YOLO("yolo11s.pt")  # Use a smaller model for faster training and decent accuracy

# Train the model with improved settings
model.train(
    data="datasets/Ziro/combined_data.yaml",  # Path to dataset configuration file
    epochs=100,  # Increase the number of epochs for better convergence
    imgsz=640,  # Image size
    batch=8,  # Increase batch size for better generalization (if memory allows)
    device="cpu",  # Use cuda for faster training (change to "cpu" if GPU is not available)
    lr0=0.01,  # Initial learning rate
    lrf=0.001,  # Final learning rate (with cosine annealing for smoother convergence)
    momentum=0.937,  # Momentum for optimizer
    weight_decay=0.0005,  # Weight decay for regularization
    warmup_epochs=3,  # Warmup epochs for learning rate
    warmup_momentum=0.8,  # Warmup momentum
    warmup_bias_lr=0.1,  # Warmup bias learning rate
    box=5.0,  # Loss weight for bounding box regression
    cls=1.0,  # Loss weight for classification
    dfl=2.0,  # Loss weight for distribution focal loss
    augment=True,  # Enable data augmentation
    hsv_h=0.015,  # Hue augmentation
    hsv_s=0.7,  # Saturation augmentation
    hsv_v=0.4,  # Value augmentation
    degrees=10.0,  # Rotation augmentation
    translate=0.1,  # Translation augmentation
    scale=0.5,  # Scale augmentation
    shear=0.0,  # Shear augmentation
    flipud=0.0,  # Flip up-down augmentation
    fliplr=0.5,  # Flip left-right augmentation
    mosaic=1.0,  # Enable mosaic augmentation with 100% probability
    mixup=0.2,  # Enable mixup augmentation with 20% probability
    copy_paste=0.2,  # Enable copy-paste augmentation with 20% probability
    erasing=0.4,  # Random erasing augmentation probability
    auto_augment="randaugment",  # Use RandAugment for auto-augmentation
    patience=20,  # Early stopping patience (stop training if no improvement for 20 epochs)
    save_period=10,  # Save model checkpoint every 10 epochs
    plots=True,  # Enable plotting of training metrics
    val=True,  # Enable validation during training
)