from projects.src.config.configuration import ConfiguartionManager
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
            train_df = pd.read_csv(self.model_config.train_path)

            X_train = train_df.drop(columns=["log_price"])
            y_train = train_df["log_price"]

            model = RandomForestRegressor()
            model = model.fit(X_train, y_train)
            joblib.dump(model, os.path.join(self.model_config.root_dir, self.model_config.model_name))


        except Exception as e:
            raise CustomException(e, sys)
        
