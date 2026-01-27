import os
import sys
import pandas as pd
import numpy as np
from projects.src.utils.logger import logging
from projects.src.utils.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.config.configuration import ConfiguartionManager
from projects.src.utils.logger import logging
from typing import List
from sklearn.model_selection import train_test_split

class DataTransformation:
    '''
        EDA
        Handle missing
        Encode categorical
        Scale numerical
        Feature engineering + Pipeline
        Save transformer v.v
    '''
    def __init__(self):
        data_transform = ConfiguartionManager()
        self.config = data_transform.get_data_transformation_config()

    def handle_missing_value(self, df: pd.DataFrame):
        try:
            missing_value = df.isnull().sum()

            if missing_value.sum() > 0:
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
        
    def features_engineering(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        num_cols: List[any], 
        cat_cols: List[any]
    ):
        try:
            # log transform target feature
            logging.info(f"Target: {target_col}")
            logging.info(f"Skewness target column before log trasform {df[target_col].skew()}")
            log_target = "log_" + target_col
            df[log_target] = np.log1p(df[target_col])
            logging.info(f"Skewness target column after log trasform {df[log_target].skew()}")

            df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)

            top_skewness = {}
            for col in num_cols:
                top_skewness[col] = df[col].skew()

            filtered = {k:v for k, v in top_skewness.items() if abs(v) > 1}
            logging.info(sorted(filtered.items(), key=lambda x : x[1], reverse=True))

            # log transform top skewness
            for col in list(filtered.keys()):
                log_col = "log_" + col
                df[log_col] = np.log1p(df[col])
                logging.info(f"Log col {log_col}: {df[log_col].skew()}, col {col}: {df[col].skew()}")

            # features selection
            cols_drop = ["id", "date", "yr_renovated", "zipcode", target_col]
            cols_drop += list(filtered.keys())
            df = df.drop(columns=cols_drop)

            return df

        except Exception as e:
            raise CustomException(e, sys)


    def init_data_transformation(self):
        try:
            logging.info("Creating data transformation...")
            path = str(self.config.data_path)
            df = pd.read_csv(path)

            self.handle_missing_value(df)
            df = self.handle_dupplidate(df)
            # df = ingestion.handle_ouliers()
            target_column = "price"
            num_columns = list((dict(self.config.num_columns)).keys())
            cat_columns = list((dict(self.config.cat_columns)).keys())
            df = self.features_engineering(df, target_column, num_columns, cat_columns)
            df.to_csv(self.config.data_transform_path)
            return df

        except Exception as e:
            raise CustomException(e, sys)
if __name__ == "__main__":

    ingestion = DataTransformation()
    ingestion.init_data_transformation()