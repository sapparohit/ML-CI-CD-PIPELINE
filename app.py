from flask import Flask, request, render_template
import sys

from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException

application = Flask(__name__)
app = application


# Home page
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        url = request.form.get("url")

        if url is None or url.strip() == "":
            return render_template(
                "home.html",
                result="Please enter a valid URL"
            )

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(url)

        # prediction will be 0 or 1
        if prediction == 1:
            result = "Phishing URL "
        else:
            result = "Legitimate URL "

        return render_template(
            "home.html",
            result=result
        )

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
