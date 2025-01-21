import os
import shutil
import random

# Path to the main folders
base_path = 'datasets/One'  # Update this path
new_base_path = 'datasets/One_2percent'  # Update this path

# List of folders
folders = ['train', 'valid', 'test']

# Create new folders
for folder in folders:
    os.makedirs(os.path.join(new_base_path, folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_base_path, folder, 'labels'), exist_ok=True)

# Copy 2% of the files
for folder in folders:
    image_folder = os.path.join(base_path, folder, 'images')
    label_folder = os.path.join(base_path, folder, 'labels')
    
    images = os.listdir(image_folder)
    num_images = len(images)
    num_to_copy = int(num_images * 0.02)
    
    selected_images = random.sample(images, num_to_copy)
    
    for img in selected_images:
        label = img.replace('.png', '.txt')  # Assuming label files are in .txt format
        label_path = os.path.join(label_folder, label)
        
        # Check if the label file exists
        if os.path.exists(label_path):
            shutil.copy(os.path.join(image_folder, img), os.path.join(new_base_path, folder, 'images', img))
            shutil.copy(label_path, os.path.join(new_base_path, folder, 'labels', label))
        else:
            print(f"Label file not found for image: {img}. Skipping...")

print("Files copied successfully.")