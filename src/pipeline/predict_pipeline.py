import sys
import os

import numpy as np

from src.exception import CustomException
from src.utils import load_object
from src.loggin import logging


class PredictPipeline:
    def __init__(self):
        self.vectorizer_path = os.path.join(
            "artifacts", "tfidf_vectorizer.pkl"
        )
        self.logistic_model_path = os.path.join(
            "artifacts", "logistic_model.pkl"
        )
        self.knn_model_path = os.path.join(
            "artifacts", "knn_model.pkl"
        )

    def predict(self, url: str, model_type: str = "logistic"):
        """
        Predict whether a URL is phishing or legitimate

        model_type:
            - "logistic" (default)
            - "knn"
        """
        try:
            logging.info("Loading vectorizer and model")

            vectorizer = load_object(self.vectorizer_path)

            if model_type == "knn":
                model = load_object(self.knn_model_path)
            else:
                model = load_object(self.logistic_model_path)

            logging.info("Transforming input URL")

            url_vec = vectorizer.transform([url])

            prediction = model.predict(url_vec)[0]

            logging.info(f"Prediction completed: {prediction}")

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
