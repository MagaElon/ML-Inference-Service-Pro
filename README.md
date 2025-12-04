# ML-Inference-Service-Pro

# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model("model.keras")

def preprocess(data):
    arr = np.array(data, dtype=float)
    return arr.reshape(1, -1)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json.get("input")
        processed = preprocess(input_data)
        prediction = model.predict(processed).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()
