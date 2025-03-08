import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load the pre-trained AnimeGAN model
model_path = "path_to_animegan_model.h5"  # Replace with your model path
model = tf.keras.models.load_model(model_path, compile=False)

# Load the input image
image_path =model_path = "C:\\Users\\manoj\\Downloads\\IMG-20240207-WA0001.jpeg"  # Replace with your model path  # Replace with your image path
input_image = cv2.imread(image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Resize and preprocess the image
input_image = cv2.resize(input_image, (256, 256))  # Resize to model input size
input_image = input_image / 127.5 - 1  # Normalize to [-1, 1]
input_image = np.expand_dims(input_image, axis=0)

# Apply AnimeGAN model
output_image = model.predict(input_image)[0]
output_image = ((output_image + 1) * 127.5).astype(np.uint8)  # Denormalize

# Save the anime-style output
output_image = Image.fromarray(output_image)
output_image.save("anime_output.png")
print("Anime-style image saved as anime_output.png")

