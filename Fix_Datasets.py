import os
import shutil
from PIL import Image

# Path to the folders containing images and labels
image_folder = 'datasets/One/valid/images'  # Path to the images folder
label_folder = 'datasets/One/valid/labels'  # Path to the labels folder

# Output folders for problematic files
output_image_folder = 'datasets/Backup_Ziro/valid/problematic_images'
output_label_folder = 'datasets/Backup_Ziro/valid/problematic_labels'

# Create output folders if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# List all images and labels
images = sorted(os.listdir(image_folder))
labels = sorted(os.listdir(label_folder))

# Function to check if a label is valid
def is_label_valid(label_path):
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            values = line.split()
            if len(values) != 5:  # If the number of values is incorrect
                return False
            # Check if values can be converted to numbers
            try:
                class_id, x_center, y_center, width, height = map(float, values)
            except ValueError:
                return False
        return True
    except Exception as e:
        print(f"Error checking label: {label_path} - Error: {e}")
        return False

# Check images and labels
for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    lbl_name = img_name.replace('.jpg', '.txt')  # Assume label files have a .txt extension
    lbl_path = os.path.join(label_folder, lbl_name)

    # Check if the label exists
    if not os.path.exists(lbl_path):
        print(f"Label for image {img_name} does not exist. Moving to problematic folder.")
        shutil.move(img_path, os.path.join(output_image_folder, img_name))
        continue

    # Check if the label is valid
    if not is_label_valid(lbl_path):
        print(f"Label for image {img_name} is corrupted. Moving to problematic folder.")
        shutil.move(img_path, os.path.join(output_image_folder, img_name))
        shutil.move(lbl_path, os.path.join(output_label_folder, lbl_name))
        continue

print("Checking and moving problematic images and labels completed.")