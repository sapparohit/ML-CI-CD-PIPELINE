import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD

from src.exception import CustomException
from src.loggin import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    logistic_model_path: str = os.path.join("artifacts", "logistic_model.pkl")
    knn_model_path: str = os.path.join("artifacts", "knn_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and testing data")

            # Unpack tuples
            X_train, y_train = train_arr
            X_test, y_test = test_arr

            # =====================================================
            # Logistic Regression (FULL TF-IDF)
            # =====================================================
            logging.info("Training Logistic Regression")

            logistic_model = LogisticRegression(
                max_iter=1000,
                n_jobs=-1
            )
            logistic_model.fit(X_train, y_train)

            y_pred_lr = logistic_model.predict(X_test)
            lr_score = accuracy_score(y_test, y_pred_lr)

            logging.info(f"Logistic Regression Score: {lr_score}")
            logging.info(
                "Logistic Regression Report:\n"
                + classification_report(y_test, y_pred_lr)
            )

            save_object(
                self.model_trainer_config.logistic_model_path,
                logistic_model
            )

            # =====================================================
            # KNN (SVD + EUCLIDEAN DISTANCE — SAFE)
            # =====================================================
            logging.info("Reducing dimensionality for KNN")

            svd = TruncatedSVD(n_components=200, random_state=42)
            X_train_knn = svd.fit_transform(X_train)
            X_test_knn = svd.transform(X_test)

            logging.info("Training KNN")

            knn_model = KNeighborsClassifier(
                n_neighbors=5,
                metric="euclidean"   # ✅ IMPORTANT CHANGE
            )

            knn_model.fit(X_train_knn, y_train)

            y_pred_knn = knn_model.predict(X_test_knn)
            knn_score = accuracy_score(y_test, y_pred_knn)

            logging.info(f"KNN Score: {knn_score}")
            logging.info(
                "KNN Classification Report:\n"
                + classification_report(y_test, y_pred_knn)
            )

            save_object(
                self.model_trainer_config.knn_model_path,
                knn_model
            )

            return {
                "LogisticRegressionScore": lr_score,
                "KNNScore": knn_score
            }

        except Exception as e:
            raise CustomException(e, sys)
