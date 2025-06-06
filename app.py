from flask import Flask, request, jsonify
import joblib
from clean_text import clean_text  # if you separated it
import os

app = Flask(__name__)

# Load model + vectorizer
model = joblib.load("phishing_classifier.pkl")
tfidf = joblib.load("tfidf.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        email_text = request.data.decode("utf-8")
        cleaned = clean_text(email_text)
        features = tfidf.transform([cleaned]).toarray()
        pred = model.predict(features)
        return jsonify({"prediction": "phishing" if pred[0] == 1 else "legitimate"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# For local or Render-style deploy
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
