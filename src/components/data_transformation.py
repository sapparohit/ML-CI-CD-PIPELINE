import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.loggin import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "tfidf_vectorizer.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates TF-IDF pipeline for URL text
        """
        try:
            tfidf = TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=50000
            )

            logging.info("TF-IDF vectorizer created successfully")

            return tfidf

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read successfully")

            target_column_name = "Label"
            text_column_name = "URL"

            if target_column_name not in train_df.columns:
                raise CustomException(
                    f"Target column '{target_column_name}' not found",
                    sys
                )

            X_train = train_df[text_column_name].astype(str)
            y_train = train_df[target_column_name]

            X_test = test_df[text_column_name].astype(str)
            y_test = test_df[target_column_name]

            vectorizer = self.get_data_transformer_object()

            logging.info("Applying TF-IDF transformation")

            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            #  KEEP SPARSE MATRICES (NO toarray)
            train_arr = (X_train_vec, y_train.values)
            test_arr = (X_test_vec, y_test.values)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=vectorizer
            )

            logging.info("TF-IDF vectorizer saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)