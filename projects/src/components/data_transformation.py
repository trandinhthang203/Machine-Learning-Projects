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
        

    def handle_dupplidate(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            n_duplicates = df.duplicated().sum()

            if n_duplicates > 0:
                df = df.drop_duplicates()
                logging.info(f"Removed {n_duplicates} duplicates")
            else:
                logging.info(f"No duplicate.")
            
            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    def handle_ouliers(self, colum, df: pd.DataFrame) -> pd.DataFrame:
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
        
    def features_engineering(self, df: pd.DataFrame):
        # 
        pass



    def init_data_transformation(self):
        try:
            logging.info("Creating data transformation...")

        except Exception as e:
            raise CustomException(e, sys)

        
if __name__ == "__main__":
    ingestion = DataTransformation()
    ingestion.handle_missing_value()