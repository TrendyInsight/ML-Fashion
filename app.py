from flask import Flask, request, jsonify
from flask_cors import CORS  # Menambahkan Flask-CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import logging

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS

# Inisialisasi model dan class_names
model = load_model('model.h5')
class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boot']

# Konfigurasi logging
logging.basicConfig(filename='app.log', level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Preprocess the image
        img_array = preprocess_image(data['image'])

        # Make predictions using the model
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))  # Convert to int

        # Ensure that the values are serializable to JSON
        result = {'class': predicted_class, 'class_name': class_names[predicted_class]}
        return jsonify(result)
    except KeyError as e:
        logging.error(f"KeyError: {e}")
        return jsonify({'error': 'Invalid JSON payload. Missing key: ' + str(e)}), 400
    except Exception as e:
        logging.error(f"Exception: {e}")
        return jsonify({'error': str(e)}), 500


def preprocess_image(image_data):
    try:
        # Decode base64 and open the image
        img_data = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_data)).resize((28, 28))

        # Convert image to grayscale and normalize
        img_array = np.mean(np.array(img), axis=-1) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        return img_array
    except Exception as e:
        logging.error(f"Exception in preprocess_image: {e}")
        raise

if __name__ == '__main__':
    app.run(port=5000)
