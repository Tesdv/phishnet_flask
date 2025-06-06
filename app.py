from flask import Flask, request, jsonify
import joblib
import numpy as np

from clean_text import clean_text  # your preprocessing function

# Load model and TF-IDF vectorizer
model = joblib.load('phishing_classifier.pkl')
tfidf = joblib.load('tfidf.pkl')  # if you saved it separately

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.data.decode('utf-8')  # Get raw string

    # Preprocess
    cleaned = clean_text(email_text)
    features = tfidf.transform([cleaned])

    # Predict
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    result = "Phishing" if prob >= 0.98 else "Legitimate"
    return jsonify({"prediction": result, "probability": prob})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
