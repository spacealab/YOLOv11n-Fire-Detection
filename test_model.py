import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def test_model_on_images(model_path, images_folder, output_folder):
    # Load the trained model
    model = YOLO(model_path)

    # Use CPU explicitly
    device = 'cpu'
    print(f"Using device: {device}")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    if not image_files:
        print("No images found in the specified folder.")
        return

    # Process images with a progress bar
    for image_name in tqdm(image_files, desc="Processing images"):
        # Load the image
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)

        # Perform object detection
        results = model(image)

        # Draw bounding boxes on the image
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes in (x1, y1, x2, y2) format
            confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                label = f"{model.names[int(class_id)]} {confidence:.2f}"  # Create label with class name and confidence

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label text

        # Save the output image
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, image)

    print(f"\nProcessed {len(image_files)} images. Results saved in '{output_folder}'.")

if __name__ == "__main__":
    # Path to the trained model
    model_path = 'runs/detect/yolov9-c-fire.pt'  # Replace with your model path

    # Path to the folder containing test images
    images_folder = 'datasets/test/images'  # Replace with your test images folder

    # Path to the folder where output images will be saved
    output_folder = 'datasets/test/output_images'  # Replace with your desired output folder

    # Test the model on images
    test_model_on_images(model_path, images_folder, output_folder)