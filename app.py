from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
CORS(app, support_credentials=True)  # Enabling CORS

# Load the pre-trained model (make sure the path is accessible)
model = load_model('mymodel.h5', compile=False)

def preprocess_image(image_path, img_width, img_height):
    """
    Preprocess the image to fit the model's input requirements.
    """
    img = load_img(image_path, target_size=(img_width, img_height), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_emotion(image_path, model, img_width, img_height):
    """
    Predicts the emotion of an image using a pre-trained model.
    """
    img_array = preprocess_image(image_path, img_width, img_height)
    predictions = model.predict(img_array)
    class_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']  # Modify as per your labels
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_labels[predicted_class]

@app.route('/')
def home():
    return "Predict server is running"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was sent
    if 'file1' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file1']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    # Save the received image
    static_folder = 'static'
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    filename = os.path.join(static_folder, 'uploaded_image.jpg')
    file.save(filename)

    # Predict emotion
    try:
        predicted_emotion = predict_emotion(filename, model, 48, 48)
        return jsonify({'emotion': predicted_emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
