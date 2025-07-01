from flask import Flask, request, jsonify
from clean_text import clean_text, extract_features
import joblib
import os

app = Flask(__name__)

model = joblib.load("phishing_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        email_text = request.data.decode("utf-8")
        features, _ = extract_features([email_text], tfidf=tfidf, fit=False)

        pred = model.predict(features)
        return jsonify({"prediction": "phishing" if pred[0] == 1 else "legitimate"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)