import numpy as np
from flask import Flask, request, render_template
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# -----------------------------
# Naive Bayes Function
# -----------------------------
def naive_bayes_predict(X, y, sample):

    model = GaussianNB()

    X = np.array(X)
    y = np.array(y)
    sample = np.array(sample).reshape(1,-1)

    model.fit(X,y)

    prediction = model.predict(sample)

    return prediction[0]


# -----------------------------
# Home Page (Frontend)
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    X = request.form["X"]
    y = request.form["y"]
    sample = request.form["sample"]

    X = [list(map(float,row.split(","))) for row in X.split(";")]
    y = list(map(int,y.split(",")))
    sample = list(map(float,sample.split(",")))

    prediction = naive_bayes_predict(X,y,sample)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)