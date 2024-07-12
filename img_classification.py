import boto3
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os

# Setup AWS S3 client
s3 = boto3.client('s3')

# S3 bucket and file details
bucket_name = 'datasetpenyakitpadi'
model_file_name = 'keras_model.h5'
labels_file_name = 'labels.txt'

# Local directory to save the downloaded files
local_dir = 'web-klasifikasi-penyakitpadi'
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

model_path = os.path.join(local_dir, model_file_name)
labels_path = os.path.join(local_dir, labels_file_name)

# Download model from S3
s3.download_file(bucket_name, model_file_name, model_path)

# Download labels from S3
s3.download_file(bucket_name, labels_file_name, labels_path)

# Load the model
model = load_model(model_path, compile=False)

# Load the labels
with open(labels_path, "r") as file:
    class_names = file.readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image_path = "<IMAGE_PATH>"  # Path to your image
image = Image.open(image_path).convert("RGB")

# Resize the image to be at least 224x224 and then crop from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predict the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name.strip(), end="")
print("Confidence Score:", confidence_score)
