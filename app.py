from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
from keras.models import load_model





app = Flask(__name__)
CORS(app, support_credentials=True)  # Enabling CORS


@app.route('/')
def home():
    return "Predict server is running"

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the static folder exists
    static_folder = 'static'
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
 
    file = request.files['file1']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    # Save the received image
    filename = os.path.join(static_folder, 'file.jpg')
    file.save(filename)

    # Process the image
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 3)
    face_detected = False

    for x, y, w, h in faces:
        face_detected = True
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cropped = img[y:y+h, x:x+w]

    # If no face is detected, use the whole image
    if not face_detected:
        cropped = img

    image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = np.reshape(image, (1, 48, 48, 1))

    # Load the model and predict
    model = load_model('model.h5', compile=False)
    prediction = model.predict(image)
    label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
    predicted_index = np.argmax(prediction)
    final_prediction = label_map[predicted_index]

    return jsonify({'emotion': final_prediction})

if __name__ == '__main__':
    app.run()
