from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    img_array = preprocess_image(data['image'])
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    result = {'class': predicted_class, 'class_name': names_info[predicted_class]}
    return jsonify(result)

def preprocess_image(image_data):
    # Implement your image preprocessing logic here
    pass

if __name__ == '__main__':
    app.run(port=5000)
