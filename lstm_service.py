from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/lstm_scaler.pkl"
SEQUENCE_LENGTH = 10

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    features = np.array(request.json["features"])
    features = scaler.transform(features)

    seq = features[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, features.shape[1])
    pred = model.predict(seq)[0][0]

    return jsonify({"lstm_score": float(pred)})

if __name__ == "__main__":
    app.run(port=6000)