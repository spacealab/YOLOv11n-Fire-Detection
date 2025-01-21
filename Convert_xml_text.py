import os
import xml.etree.ElementTree as ET

def convert_annotation(xml_file, output_file, class_id):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    for obj in root.iter('object'):
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Calculate center coordinates and dimensions
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        # Write to output file
        output_file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

def convert_xml_to_yolo(input_dir, output_dir, class_id):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(input_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(input_dir, xml_file)
            output_path = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))

            with open(output_path, 'w') as output_file:
                convert_annotation(xml_path, output_file, class_id)

# Example usage
input_directory = 'Datasets/N/Annotations'
output_directory = 'Datasets/N'
class_id = 0  # Replace with the appropriate class ID for your dataset

convert_xml_to_yolo(input_directory, output_directory, class_id)