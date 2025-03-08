from ultralytics import YOLO
import cv2
from moviepy import ImageClip, CompositeVideoClip
import numpy as np

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use pre-trained YOLOv8 model

# Input image and output video paths
image_path = "C:\\Users\\manoj\\Downloads\\IMG-20240207-WA0001.jpeg"
output_video = "C:\\Users\\manoj\\Videos\\motion_video.avi"
# Perform object detection
results = model(image_path)
detections = results[0].boxes.xyxy.numpy()  # Bounding box coordinates (x1, y1, x2, y2)
labels = results[0].boxes.cls.numpy()  # Class labels
scores = results[0].boxes.conf.numpy()  # Confidence scores
# Read the original image
image = cv2.imread(image_path)
height, width, _ = image.shape

# Prepare layers for each detected object
object_clips = []
for idx, bbox in enumerate(detections):
    x1, y1, x2, y2 = bbox.astype(int)
    cropped_object = image[y1:y2, x1:x2]
    obj_clip = ImageClip(cropped_object[:, :, ::-1])  # Convert BGR to RGB
    obj_clip =obj_clip.with_duration(9)  # Duration of animation
    # Animate object (moving in a circle)
    obj_clip = obj_clip.with_position(lambda t: (100 + 100 * t, 100 + 50 * t))  # Animation path
    object_clips.append(obj_clip)

# Background (static image)
background = ImageClip(image_path).with_duration(5)

# Combine background and object animations
final_video = CompositeVideoClip([background] + object_clips)

# Export the video
final_video.write_videofile("output_video.mp4", codec="libx264", fps=24)

print(f"Video saved at: {output_video}")