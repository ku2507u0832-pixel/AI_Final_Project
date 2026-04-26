from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Train model directly (no pkl files needed!)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
           'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv(url, names=columns)

# Clean data
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].mean())

# Train
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained and ready!")

@app.route('/')
def home():
    return jsonify({"status": "Diabetes Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
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
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] * 100

        return jsonify({
            "prediction": int(prediction),
            "result": "Diabetic" if prediction == 1 else "Not Diabetic",
            "probability": round(probability, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
