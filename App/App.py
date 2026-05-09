from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

# Load model once
model = load_model("../model/90acc.keras")

if model is not None:
    print("Model is available")
else:
    print("Model not found")

labels = {0: "Cat", 1: "Dog"}


def preprocess_image(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = preprocess_image(file)

    prediction = model.predict(img)[0][0]

    if prediction >= 0.5:
        predicted_class = 1
        confidence = float(prediction)
    else:
        predicted_class = 0
        confidence = float(1 - prediction)

    return jsonify({"class": labels[predicted_class], "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)
    
