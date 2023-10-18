import requests
import json

url = "http://127.0.0.1:5000/predict"  # Ganti dengan URL API Anda

# Contoh data gambar dalam bentuk base64 (atau sesuaikan dengan format yang diterima oleh API Anda)
image_data = "base64_encoded_image_data"

data = {"image": image_data}

# Mengirim permintaan POST ke API
response = requests.post(url, json=data)

# Mendapatkan hasil prediksi dari respons
result = json.loads(response.text)
print(result)
