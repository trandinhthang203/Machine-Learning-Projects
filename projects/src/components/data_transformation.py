import os
import sys
import pandas as pd
import numpy as np
from projects.src.utils.logger import logging
from projects.src.utils.exception import CustomException
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.config.configuration import ConfiguartionManager
from projects.src.utils.logger import logging
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class SkewnessLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, skewness_thresold = 1.0):
        self.skewness_thresold = skewness_thresold
        self.transform_cols_ = []

    # học từ tập train
    def fit(self, X, y = None):
        if isinstance(X, pd.DataFrame):
            X_numerical = X.select_dtypes(include="number")
            skewness = X_numerical.skew()
            self.transform_cols_ = skewness[abs(skewness) > self.skewness_thresold].index.to_list()

        return self

    # trainform ở tập train/test
    def transform(self, X):
        # dùng x_copy để không thay đổi dữ liệu gốc
        # trong trường hợp dùng trong pipeline sẽ bị thay đổi dữ liệu ban đầu
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            for col in self.transform_cols_:
                X_copy[f"log_{col}"] = np.log1p(X_copy[col])
                logging.info(f"Log-transformed '{col}' → skewness: {X_copy[f'log_{col}'].skew():.3f}")

        return X_copy


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        if isinstance(X_copy, pd.DataFrame):
            X_copy["is_renovated"] = (X_copy["yr_renovated"] > 0).astype(int)
            cols_drop = ["id", "date", "yr_renovated", "zipcode"]
            X_copy.drop(columns=cols_drop)

        return X_copy
    
class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()




class DataTransformation:
    '''
        EDA
        Cleaning
        Transform
        Feature engineer
        Feature selection
        Create pipeline
    '''
    def __init__(self):
        data_transform = ConfiguartionManager()
        self.config = data_transform.get_data_transformation_config()
        self.preprocessor = None

    def _create_preprocessing_pipeline(self, num_cols, cat_cols):
        
        numerical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", OrdinalEncoder())
            ]
        )

        processor = ColumnTransformer(
            [
                ("num", numerical_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols)
            ],
            remainder="passthrough"
        )

        full_pipeline = Pipeline(
            [
                ("feature_engineer", FeatureEngineering()),
                ("skewness_transform", SkewnessLogTransformer()),
                ("processor", processor)
            ]
        )

        return full_pipeline
        

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