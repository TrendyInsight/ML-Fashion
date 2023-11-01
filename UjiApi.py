import requests
import base64
from PIL import Image
import json

# Load an example image for testing
image_path = "C:/Users/param/OneDrive/Dokumen/Gum/dataset/images_compressed/1d887da7-8102-488b-b0f0-ff8c9d507926.jpg"

# Convert the image to base64 encoding
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Prepare the request payload
payload = {"image": encoded_image}
headers = {"Content-Type": "application/json"}

# Make a POST request to the Flask server
response = requests.post("http://127.0.0.1:5000/predict", json=payload, headers=headers)

# Check the response
if response.status_code == 200:
    result = response.json()
    if 'class' in result:  # Add this line to check if 'class' key is present
        print("Predicted class:", result['class'])
        print("Class name:", result['class_name'])
    else:
        print("Error: 'class' key not present in the response.")
else:
    print("Error:", response.status_code, response.text)
