from src.config.configuration import ConfiguartionManager
from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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
            X = df.drop(columns=["log_price"])
            y = df["log_price"]

            num_columns = [x for x in list(df.columns) if (x not in self.model_config.cat_columns) and (x != "log_price")]
            logging.info(f"Numerical columns: {num_columns}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

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

            trainer_pipeline.fit(X_train, y_train)
            y_pred = trainer_pipeline.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"MSE: {mse}, MAE: {mae}, R2:{r2}")


        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":

    ingestion = ModelTrainer()
    ingestion.init_model_traner()