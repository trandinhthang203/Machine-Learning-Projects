from src.config.configuration import ConfiguartionManager
from projects.src.utils.logger import logging
from projects.src.utils.exception import CustomException
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class ModelTrainer:
    def __init__(self):
        config = ConfiguartionManager()
        self.model_config = config.get_model_trainer_config()

    def init_model_traner(self):
        try:
            data_path = self.model_config.data_path
            test_size = self.model_config.test_size
            df = pd.read_csv(data_path, index_col=0)
            logging.info(f"Columns: {df.columns}")

            num_columns = [x for x in list(df.columns) if (x not in self.model_config.cat_columns) and (x != "log_price")]
            logging.info(f"Numerical columns: {num_columns}")

            data_train, data_test = train_test_split(df, test_size=test_size, random_state=42)
            data_train.to_csv(self.model_config.train_path)
            data_test.to_csv(self.model_config.test_path)

            X_train = df.drop(columns=["log_price"])
            y_train = df["log_price"]


            num_pipeline = Pipeline(
                [
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                [
                    ("encoder", OrdinalEncoder())
                ]
            )

            processor = ColumnTransformer(
                [
                    ("num", num_pipeline, num_columns),
                    # ("cat", cat_pipeline, self.model_config.cat_columns),
                ],
                remainder="passthrough"
            )

            trainer_pipeline = Pipeline(
                [
                    ("processor", processor),
                    ("regressor", RandomForestRegressor())
                ]
            )

            model = trainer_pipeline.fit(X_train, y_train)
            joblib.dump(model, os.path.join(self.model_config.root_dir, self.model_config.model_name))


        except Exception as e:
            raise CustomException(e, sys)
        
