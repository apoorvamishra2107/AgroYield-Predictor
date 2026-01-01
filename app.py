import numpy as np
from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.array([float_features])
    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction_text=f"The Predicted Crop is: {prediction}"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
