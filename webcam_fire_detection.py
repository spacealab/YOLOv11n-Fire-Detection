from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO("runs/detect/train4/weights/best.pt")  # Path to the trained model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 indicates the default webcam

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Process the frame with the YOLO model
    results = model(frame)

    # Display the results on the frame
    annotated_frame = results[0].plot()  # Draws bounding boxes and labels on the frame

    # Convert the frame from BGR to RGB (matplotlib uses RGB)
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Show the processed frame using matplotlib
    plt.imshow(annotated_frame_rgb)
    plt.axis('off')  # Hide axes
    plt.pause(0.01)  # Pause to update the plot
    plt.clf()  # Clear the current figure

    # If 'q' is pressed, stop the loop
    if plt.waitforbuttonpress(0.01):
        break

# Release the webcam and close all windows
cap.release()
plt.close()