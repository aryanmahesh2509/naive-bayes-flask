import numpy as np
from flask import Flask, request, jsonify
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# -----------------------------
# Naive Bayes Classifier
# -----------------------------
def naive_bayes_classify(X, y, sample):

    model = GaussianNB()

    X = np.array(X)
    y = np.array(y)
    sample = np.array(sample).reshape(1, -1)

    model.fit(X, y)

    prediction = model.predict(sample)

    return prediction[0]


# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return "Naive Bayes Flask API Working"


# -----------------------------
# API Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def run_nb():

    data = request.json

    X = data["X"]       # training features
    y = data["y"]       # labels
    sample = data["sample"]  # new input

    prediction = naive_bayes_classify(X, y, sample)

    return jsonify({
        "prediction": int(prediction)
    })


if __name__ == "__main__":
    app.run(debug=True)