import cv2
import numpy as np
import streamlit as st

# Load YOLOv3 weights and configuration
net = cv2.dnn.readNet("wget https://pjreddie.com/media/files/yolov3.weights", "yolov3.cfg")

# Load COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize colors for bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Streamlit title
st.title("Real-time Object Detection using YOLOv3")

# Streamlit sidebar for user control
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.4)

# Start webcam capture
cap = cv2.VideoCapture(0)

stframe = st.empty()  # Placeholder for the video frame

# Main detection loop
while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()

    # Check if frame is valid
    if not ret:
        st.error("Failed to capture frame from webcam.")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

    # Convert frame to blob for YOLOv3 input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set input for YOLOv3 network
    net.setInput(blob)

    # Forward pass through network
    outs = net.forward(output_layers)

    # Initialize lists to store bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Process each output layer
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                # Object detected, get bounding box coordinates
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Draw bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    # Convert frame (BGR to RGB) for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display frame in Streamlit
    stframe.image(frame_rgb, channels="RGB")

    # Break loop if Streamlit session ends
    if st.sidebar.button("Stop"):
        break

# Release video capture and close all windows
cap.release()
