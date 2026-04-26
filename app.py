from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow frontend to talk to backend

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return jsonify({ "status": "Diabetes Prediction API is running!" })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract all 8 features from request
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodpressure']),
            float(data['skinthickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetespedigree']),
            float(data['age'])
        ]

        # Scale and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] * 100

        return jsonify({
            "prediction": int(prediction),
            "result": "Diabetic" if prediction == 1 else "Not Diabetic",
            "probability": round(probability, 1)
        })

    except Exception as e:
        return jsonify({ "error": str(e) }), 400

if __name__ == '__main__':
    app.run(debug=True)
