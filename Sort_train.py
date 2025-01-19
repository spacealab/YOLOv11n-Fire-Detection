import os
import shutil

# Paths to folders containing images and labels
input_folders = ['datasets/One/valid', 'datasets/Ziro/valid']  # Input folders
output_image_folder = 'datasets/Exports/valid/images'  # Output folder for images
output_label_folder = 'datasets/Exports/valid/labels'  # Output folder for labels

# Create output folders if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# Counter for new names
counter = 1

# Loop through input folders
for folder in input_folders:
    image_folder = os.path.join(folder, 'images')  # Path to images folder
    label_folder = os.path.join(folder, 'labels')  # Path to labels folder
    
    # List image and label files
    image_files = sorted(os.listdir(image_folder))
    label_files = sorted(os.listdir(label_folder))
    
    # Loop to copy and rename files
    for image_file, label_file in zip(image_files, label_files):
        # New names for files
        new_image_name = f'{counter:04d}.jpg'  # New name format for images
        new_label_name = f'{counter:04d}.txt'  # New name format for labels
        
        # Full paths for old and new files
        old_image_path = os.path.join(image_folder, image_file)
        old_label_path = os.path.join(label_folder, label_file)
        new_image_path = os.path.join(output_image_folder, new_image_name)
        new_label_path = os.path.join(output_label_folder, new_label_name)
        
        # Copy and rename files
        shutil.copy(old_image_path, new_image_path)
        shutil.copy(old_label_path, new_label_path)
        
        # Increment counter
        counter += 1

print(f'{counter-1} files successfully merged and renamed.')