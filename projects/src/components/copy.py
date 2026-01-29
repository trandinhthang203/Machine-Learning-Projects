import os
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from projects.src.utils.logger import logging
from projects.src.utils.exception import CustomException
from src.config.configuration import ConfiguartionManager
import joblib

# ============= CUSTOM TRANSFORMERS =============

class SkewnessLogTransformer(BaseEstimator, TransformerMixin):
    """Apply log transform to skewed features"""
    
    def __init__(self, skewness_threshold=1.0):
        self.skewness_threshold = skewness_threshold
        self.transform_cols_ = []
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            skewness = X.skew()
            self.transform_cols_ = skewness[abs(skewness) > self.skewness_threshold].index.tolist()
            logging.info(f"Identified {len(self.transform_cols_)} skewed columns: {self.transform_cols_}")
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            for col in self.transform_cols_:
                if col in X_copy.columns:
                    X_copy[f'log_{col}'] = np.log1p(X_copy[col])
                    logging.info(f"Log-transformed '{col}' → skewness: {X_copy[f'log_{col}'].skew():.3f}")
        return X_copy


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create engineered features"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if isinstance(X_copy, pd.DataFrame):
            # Renovation flag
            if 'yr_renovated' in X_copy.columns:
                X_copy['is_renovated'] = (X_copy['yr_renovated'] > 0).astype(int)
                logging.info("Created 'is_renovated' feature")
            
            # House age
            if 'yr_built' in X_copy.columns:
                current_year = pd.Timestamp.now().year
                X_copy['house_age'] = current_year - X_copy['yr_built']
                logging.info("Created 'house_age' feature")
            
            # Price per sqft (if available)
            if 'sqft_living' in X_copy.columns and 'sqft_lot' in X_copy.columns:
                X_copy['living_lot_ratio'] = X_copy['sqft_living'] / (X_copy['sqft_lot'] + 1)
                logging.info("Created 'living_lot_ratio' feature")
        
        return X_copy


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers using IQR method"""
    
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in X.select_dtypes(include=[np.number]).columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                
                self.bounds_[col] = {
                    'lower': q1 - self.multiplier * iqr,
                    'upper': q3 + self.multiplier * iqr
                }
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if isinstance(X_copy, pd.DataFrame):
            for col, bounds in self.bounds_.items():
                if col in X_copy.columns:
                    n_outliers = ((X_copy[col] < bounds['lower']) | 
                                 (X_copy[col] > bounds['upper'])).sum()
                    
                    X_copy[col] = X_copy[col].clip(bounds['lower'], bounds['upper'])
                    
                    if n_outliers > 0:
                        logging.info(f"Clipped {n_outliers} outliers in '{col}'")
        
        return X_copy


# ============= MAIN TRANSFORMATION CLASS =============

class DataTransformation:
    def __init__(self):
        data_transform = ConfiguartionManager()
        self.config = data_transform.get_data_transformation_config()
        self.preprocessor = None

    def _create_preprocessing_pipeline(self, num_cols, cat_cols):
        """Create complete preprocessing pipeline"""
        
        # Numerical pipeline
        num_pipeline = Pipeline([
            ('outlier_clipper', OutlierClipper(multiplier=1.5)),
            ('scaler', StandardScaler())
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, num_cols)
            ],
            remainder='passthrough'  # Keep other columns
        )
        
        # Full pipeline with feature engineering
        full_pipeline = Pipeline([
            ('feature_engineer', FeatureEngineer()),
            ('skewness_transform', SkewnessLogTransformer(skewness_threshold=1.0)),
            ('preprocessor', preprocessor)
        ])
        
        return full_pipeline

    def init_data_transformation(self):
        """Main transformation orchestrator"""
        logging.info("=" * 50)
        logging.info("DATA TRANSFORMATION STARTING")
        logging.info("=" * 50)
        
        try:
            # Load data
            df = pd.read_csv(str(self.config.data_path))
            logging.info(f"Loaded data: {df.shape}")
            
            # Handle duplicates
            n_duplicates = df.duplicated().sum()
            if n_duplicates > 0:
                df = df.drop_duplicates()
                logging.info(f"Removed {n_duplicates} duplicate rows")
            
            # Get column lists
            target_col = "price"
            num_cols = list(dict(self.config.num_columns).keys())
            cat_cols = list(dict(self.config.cat_columns).keys())
            
            # Log transform target
            if target_col in df.columns:
                df['log_price'] = np.log1p(df[target_col])
                logging.info(f"Target skewness: {df[target_col].skew():.3f} → {df['log_price'].skew():.3f}")
            
            # Split features and target
            drop_cols = ['id', 'date', target_col] if 'id' in df.columns else ['date', target_col]
            X = df.drop(columns=drop_cols, errors='ignore')
            y = df['log_price']
            
            # Create and fit pipeline
            self.preprocessor = self._create_preprocessing_pipeline(num_cols, cat_cols)
            
            # Train-test split BEFORE fitting (avoid data leakage)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Fit on training data only
            self.preprocessor.fit(X_train, y_train)
            logging.info("Pipeline fitted on training data")
            
            # Transform both sets
            X_train_transformed = self.preprocessor.transform(X_train)
            X_test_transformed = self.preprocessor.transform(X_test)
            
            # Save transformed data
            train_df = pd.DataFrame(X_train_transformed)
            train_df['target'] = y_train.values
            test_df = pd.DataFrame(X_test_transformed)
            test_df['target'] = y_test.values
            
            train_path = str(self.config.data_transform_path).replace('.csv', '_train.csv')
            test_path = str(self.config.data_transform_path).replace('.csv', '_test.csv')
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            # Save preprocessor
            preprocessor_path = os.path.join(self.config.root_dir, 'preprocessor.joblib')
            joblib.dump(self.preprocessor, preprocessor_path)
            logging.info(f"Preprocessor saved: {preprocessor_path}")
            
            logging.info("=" * 50)
            logging.info("DATA TRANSFORMATION COMPLETED")
            logging.info(f"Train shape: {train_df.shape}")
            logging.info(f"Test shape: {test_df.shape}")
            logging.info("=" * 50)
            
            return train_path, test_path, preprocessor_path

        except Exception as e:
            logging.error(f"Transformation failed: {str(e)}")
            raise CustomException(e, sys)
