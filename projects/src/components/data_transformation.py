import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.config.configuration import ConfiguartionManager
from src.logger import logging


class DataTransformation:
    '''
        EDA
        Handle missing
        Encode categorical
        Scale numerical
        Feature engineering
        Save transformer v.v
    '''
    def __init__(self):
        data_transform = ConfiguartionManager()
        self.config = data_transform.get_data_transformation_config()

    def handle_missing_value(self, df: pd.DataFrame):
        try:
            missing_value = df.isnull().sum()

            if missing_value > 0:

                logging.info(f"Total missing value {missing_value[missing_value > 0]}")
            else:
                logging.info(f"No missing value.")

            # countinue

        except Exception as e:
            raise CustomException(e, sys)
        

    def handle_dupplidate(self, df: pd.DataFrame):
        try:
            n_duplicates = df.duplicated().sum()

            if n_duplicates > 0:
                df.drop_duplicates()
                logging.info(f"Removed {n_duplicates} duplicates")
            else:
                logging.info(f"No duplicate.")

        except Exception as e:
            raise CustomException(e, sys)
    def get_data_transformation_obj(self):
        logging.info("Creating data transformation...")
        
        numerical_columns = ['bathrooms',
                                'floors',
                                'yr_built',
                                'lat',
                                'long',
                                'is_renovated',
                                'log_price',
                                'log_sqft_lot',
                                'log_sqft_lot15',
                                'log_bedrooms',
                                'log_sqft_basement',
                                'log_sqft_living',
                                'log_sqft_above',
                                'log_sqft_living15']
        
        try:
            num_pipeline = Pipeline(
                [
                    # ("imputer", SimpleImputer(strategy = "mean"))
                    ("sacler", StandardScaler())
                ]
            )

            processer = ColumnTransformer(
                [
                    ("encoder", num_pipeline, numerical_columns)
                ]
            )
            logging.info("Created data transformation")
            return processer

        except Exception as e:
            raise CustomException(e, sys)

    def handle_ouliers(self, colum, df: pd.DataFrame):
        try:
            q1 = df[colum].quantile(0.25)
            q3 = df[colum].quantile(0.75)

            iqr = q3 - q1

            upper_limit = q3 + 1.5*iqr
            lower_linit = q1 - 1.5*iqr

            df.loc[(df[colum] > upper_limit), colum] = upper_limit
            df.loc[(df[colum] < lower_linit), colum] = lower_linit

            return df

        except Exception as e:
            raise CustomException(e, sys)

    
    def init_data_transformation_obj(self, train_path, test_path, target_feature = "price"):
        logging.info("Initing data transformation...")

        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            X_train = df_train.drop(columns=[target_feature])
            y_train = df_train[target_feature]

            X_test = df_test.drop(columns=[target_feature])
            y_test = df_test[target_feature]

            processor = self.get_data_transformation_obj()

            pipline = Pipeline(
                [
                    ("")
                ]
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    ingestion = DataTransformation()
    ingestion.handle_missing_value()