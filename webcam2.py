from ultralytics import YOLO
import depthai as dai
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLO model
model = YOLO("runs/detect/train8/weights/best.pt")  # Path to the trained model

# Create DepthAI pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)  # Set the preview size
cam_rgb.setInterleaved(False)  # Set planar layout

# Create an output stream
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Connect to the DepthAI device and start the pipeline
with dai.Device(pipeline) as device:
    # Output queue to get the RGB frames
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        # Get the frame from the DepthAI camera
        in_rgb = q_rgb.get()
        frame = in_rgb.getCvFrame()  # Convert to OpenCV frame

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

# Release resources
plt.close()