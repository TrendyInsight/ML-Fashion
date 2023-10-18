from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')

# Define class names
class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boot']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Preprocess the image
    img_array = preprocess_image(data['image'])

    # Make predictions using the model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_name = class_names[predicted_class]

    result = {'class': predicted_class, 'class_name': class_name}
    return jsonify(result)

def preprocess_image(image_data):
    # Convert base64 image data to a NumPy array
    img_array = np.array(Image.open(image_data).resize((28, 28)))

    # Convert image to grayscale and normalize
    img_array = np.mean(img_array, axis=-1) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    return img_array

if __name__ == '__main__':
    app.run(port=5000)
