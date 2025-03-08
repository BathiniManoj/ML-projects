import cv2
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import joblib  # For loading models

# Load trained ML models
clf = joblib.load("density_classifier.pkl")
reg = joblib.load("clearance_time_predictor.pkl")

# YOLO Paths
folder_path = r"C:\\Users\\manoj\\mlproject\\csvfile\\images"
cfg_path = "C:\\Users\\manoj\\Downloads\\yolov3.cfg"
weights_path = "C:\\Users\\manoj\\Downloads\\yolov3.weights"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Vehicle class IDs (COCO dataset: car, motorbike, bus, truck)
vehicle_class_ids = [2, 3, 5, 7]

# Process images
image_files = os.listdir(folder_path)

for image_file in image_files:
    if image_file.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not load {image_file}")
            continue

        height, width, _ = image.shape
        frame_area = height * width  # Total frame area
        avg_speed_mps = 5  # Estimated speed (modify if you have real-time data)

        # Preprocess image for YOLO
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        vehicle_count = 0
        boxes, confidences, class_ids = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_id in vehicle_class_ids:
                    vehicle_count += 1  # Count detected vehicles

        # Predict density category
        density_category = clf.predict([[vehicle_count, frame_area, avg_speed_mps]])[0]

        # Predict clearance time
        predicted_time = reg.predict([[vehicle_count, avg_speed_mps]])[0]

        # Display results on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"Vehicles: {vehicle_count}", (50, 40), font, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"Density: {density_category}", (50, 70), font, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"Clear Time: {round(predicted_time, 2)} sec", (50, 100), font, 0.8, (0, 255, 0), 2)

        # Show image
        cv2.imshow("Traffic Analysis", image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
