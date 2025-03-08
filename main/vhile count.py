import cv2
import os
import numpy as np

# Folder path where the images are stored
folder_path = r"C:\\Users\\manoj\\mlproject\\csvfile\\images"  

# YOLOv3 config and weights file paths
cfg_path = "C:\\Users\\manoj\\Downloads\\yolov3.cfg"
weights_path = "C:\\Users\\manoj\\Downloads\\yolov3.weights"

# Load the YOLO model
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Load class labels (COCO dataset labels for YOLOv3)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# List all files in the folder
image_files = os.listdir(folder_path)

# Vehicle class IDs in the COCO dataset (car, motorbike, bus, truck)
vehicle_class_ids = [2, 3, 5, 7]  

# Assumed values for traffic estimation
lane_width_m = 3.5  # Approximate lane width in meters
vehicle_length_m = 4.5  # Average vehicle length in meters
avg_speed_kmph = 20  # Estimated average speed in km/h
avg_speed_mps = avg_speed_kmph * (1000 / 3600)  # Convert km/h to m/s

# Loop through each file in the folder
for image_file in image_files:
    if image_file.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(folder_path, image_file)  
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not load {image_file}")
            continue

        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []
        vehicle_count = 0  

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)  
                confidence = scores[class_id]  

                if confidence > 0.5:
                    center_x, center_y, width, height = (
                        int(detection[0] * image.shape[1]),
                        int(detection[1] * image.shape[0]),
                        int(detection[2] * image.shape[1]),
                        int(detection[3] * image.shape[0])
                    )
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                
                if class_id in vehicle_class_ids:
                    vehicle_count += 1  
                    color = (0, 255, 0)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, str(vehicle_count), (x + int(w / 2), y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # **Traffic Density Calculation**
        frame_area = image.shape[0] * image.shape[1]  
        density = vehicle_count / frame_area  

        # **Time to Clear Traffic Calculation**
        road_length_m = (vehicle_count * vehicle_length_m)  
        clearance_time_s = road_length_m / avg_speed_mps  

        # Display information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"Vehicles: {vehicle_count}", (50, 30), font, 0.9, (0, 255, 0), 2)
        cv2.putText(image, f"Density: {round(density, 6)}", (50, 60), font, 0.9, (0, 255, 0), 2)
        cv2.putText(image, f"Clear Time: {round(clearance_time_s, 2)} sec", (50, 90), font, 0.9, (0, 255, 0), 2)

        print(f"{image_file}: {vehicle_count} vehicles, Density: {round(density, 6)}, Time to Clear: {round(clearance_time_s, 2)} sec")

        # Show Image (optional)
        cv2.imshow("Traffic Analysis", image)
        cv2.waitKey(0)

        # Save the processed image
        output_folder = os.path.join(folder_path, "output")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)

# Close OpenCV windows
cv2.destroyAllWindows()
