import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model



# Define class names
names_info = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the input image
def preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.reshape(img_array, (1, 28, 28))  # Reshape for model input
    return img_array

# Load the pre-trained model
model = load_model('model.h5')

# Function to make a prediction and update the result label
def predict_image(file_path):
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    result_label.config(text=f'Predicted Class: {names_info[predicted_class]}')

# Function to open a file dialog and get the file path
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display the selected image
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        img_label.configure(image=img)
        img_label.image = img

        # Make a prediction
        predict_image(file_path)

# GUI setup
root = tk.Tk()
root.title('Image Classification GUI')

# Create widgets
open_button = tk.Button(root, text='Open Image', command=open_file_dialog)
img_label = tk.Label(root)
result_label = tk.Label(root, text='Predicted Class:')

# Grid layout
open_button.grid(row=0, column=0, pady=10)
img_label.grid(row=1, column=0)
result_label.grid(row=2, column=0, pady=10)

# Run the GUI
root.mainloop()
