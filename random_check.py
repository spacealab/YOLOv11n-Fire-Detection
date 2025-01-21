import os
import random
from PIL import Image, ImageDraw, ImageFont

# Path to the folders containing images and labels
image_folder = 'datasets/One/test/images'  # Path to the images folder
label_folder = 'datasets/One/test/labels'  # Path to the labels folder

# List all images and labels
images = sorted(os.listdir(image_folder))
labels = sorted(os.listdir(label_folder))

# Number of random samples to check
num_samples = 100

# Randomly select 50 images and their corresponding labels
random_samples = random.sample(list(zip(images, labels)), num_samples)

# Font for displaying text on images (optional)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

# Output folder to save checked images
output_folder = 'datasets/train'
os.makedirs(output_folder, exist_ok=True)

for img_name, lbl_name in random_samples:
    img_path = os.path.join(image_folder, img_name)
    lbl_path = os.path.join(label_folder, lbl_name)

    # Load the image
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    # Load the labels
    with open(lbl_path, 'r') as f:
        lines = f.readlines()

    # Display labels on the image
    for line in lines:
        line = line.strip()  # Remove extra spaces and empty lines
        if not line:  # Skip if the line is empty
            continue

        # Split the line into values
        values = line.split()
        if len(values) != 5:  # If the number of values is incorrect, print an error
            print(f"Format error in label file: {lbl_path} - Line: {line}")
            continue

        try:
            # Convert values to numbers
            class_id, x_center, y_center, width, height = map(float, values)
        except ValueError as e:
            print(f"Error converting values to numbers in label file: {lbl_path} - Line: {line} - Error: {e}")
            continue

        # Convert normalized coordinates to pixel coordinates
        img_width, img_height = img.size
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Calculate bounding box coordinates
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Draw bounding box on the image
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

        # Display class on the image (optional)
        draw.text((x_min, y_min - 20), f"Class: {int(class_id)}", fill="red", font=font)

    # Save the image with labels in the output folder
    output_path = os.path.join(output_folder, img_name)
    img.save(output_path)

    print(f"Saved checked image: {output_path}")

print("Random check of 50 images and labels completed.")