import cv2
import urllib.request
import numpy as np
from imutils.video import FPS

# IP camera stream URL
url = 'http://192.168.84.204/capture'

# Load pre-trained model for object detection
prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Define the confidence threshold for object detection
confidence_threshold = 0.2

# Initialize the FPS counter
fps = FPS().start()

# Define the class labels for object detection
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Assign random colors to each class label
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Main loop for accessing camera stream and performing object detection
while True:
    try:
        # Access the camera stream
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        # Resize the frame to have a maximum width of 400 pixels
        frame = cv2.resize(frame, (400, 300))

        # Perform object detection
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # Display the frame
        cv2.imshow("Frame", frame)

        # Check for key press 'q' to exit the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Update the FPS counter
        fps.update()
    
    except Exception as e:
        print("Error:", e)
        break

# Stop the timer and display FPS information
fps.stop()
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()