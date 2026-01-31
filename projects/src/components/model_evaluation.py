from projects.src.config.configuration import ConfiguartionManager
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from projects.src.utils.logger import logging
from projects.src.utils.exception import CustomException
import sys
import json
import os

class ModelEvaluation:
    def __init__(self):
        model_config = ConfiguartionManager()
        self.config = model_config.get_model_evaluation_config()

    def init_model_evaluation(self):
        try:
            # load model
            model = joblib.load(self.config.model_path)
            df = pd.read_csv(self.config.test_path)
            X_test = df.drop(columns=["log_price"])
            y_test = df["log_price"]

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                "MSE": mse,
                "MAE": mae,
                "R2": r2
            }

            metrics_path = os.path.join(self.config.root_dir, self.config.metric_file_name)
            with open(metrics_path, "w", encoding="utf-8") as file:
                json.dump(metrics, file, ensure_ascii=False, indent=4)
                logging.info(f"Metrics successfuly save to {self.config.metric_file_name}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":

    ingestion = ModelEvaluation()
    ingestion.init_model_evaluation()