"""
IMPROVED HOUSE PRICE PREDICTION PROJECT
=======================================
Author: Senior ML Engineer
Purpose: Production-ready code với best practices

Improvements:
- Comprehensive data cleaning
- Advanced feature engineering
- Hyperparameter tuning
- Proper error analysis
- Modular code structure
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from scipy.stats import skew
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')


class HousePricePredictor:
    """
    A comprehensive house price prediction system with best practices.
    
    Features:
    - Automated data cleaning
    - Feature engineering
    - Model training with hyperparameter tuning
    - Comprehensive evaluation
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        
    def load_data(self, filepath):
        """Load and perform initial data inspection."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        df = pd.read_csv(filepath)
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        
        return df
    
    def clean_data(self, df):
        """Comprehensive data cleaning."""
        print("\n" + "=" * 80)
        print("DATA CLEANING")
        print("=" * 80)
        
        # Check missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing values found:")
            print(missing[missing > 0])
        else:
            print("\n✓ No missing values")
        
        # Check duplicates
        n_duplicates = df.duplicated().sum()
        print(f"\nDuplicates: {n_duplicates}")
        if n_duplicates > 0:
            df = df.drop_duplicates()
            print(f"✓ Removed {n_duplicates} duplicates")
        
        # Remove outliers using IQR method for key features
        print("\n" + "-" * 40)
        print("Removing outliers...")
        initial_count = len(df)
        
        # Price outliers (extreme luxury homes)
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[df['price'] <= Q3 + 3 * IQR]
        
        # Physical features outliers
        df = df[df['bedrooms'] < 10]  # Reasonable upper limit
        df = df[df['bathrooms'] < 8]
        df = df[df['sqft_living'] < 13000]
        df = df[df['sqft_lot'] < 1000000]
        
        removed = initial_count - len(df)
        print(f"✓ Removed {removed} outlier records ({removed/initial_count*100:.2f}%)")
        print(f"✓ Final dataset size: {len(df)}")
        
        return df
    
    def engineer_features(self, df):
        """Create meaningful features from existing ones."""
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)
        
        # Drop ID (no predictive value)
        df = df.drop(columns=['id'])
        
        # === TIME-BASED FEATURES ===
        df['date'] = pd.to_datetime(df['date'])
        df['year_sold'] = df['date'].dt.year
        df['month_sold'] = df['date'].dt.month
        df['quarter_sold'] = df['date'].dt.quarter
        
        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        df['season'] = df['month_sold'].apply(lambda x: (x % 12 + 3) // 3)
        
        # Is it peak season for house sales? (Spring/Summer)
        df['is_peak_season'] = df['season'].isin([2, 3]).astype(int)
        
        df = df.drop(columns=['date'])
        print("✓ Created time-based features: year_sold, month_sold, quarter_sold, season, is_peak_season")
        
        # === AGE FEATURES ===
        current_year = df['year_sold'].max()
        df['house_age'] = current_year - df['yr_built']
        df['is_new_house'] = (df['house_age'] <= 5).astype(int)
        
        # Renovation features
        df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
        df['years_since_renovation'] = np.where(
            df['yr_renovated'] > 0,
            current_year - df['yr_renovated'],
            df['house_age']
        )
        df['renovation_age_ratio'] = df['years_since_renovation'] / (df['house_age'] + 1)
        
        print("✓ Created age features: house_age, is_new_house, is_renovated, years_since_renovation")
        
        # === RATIO FEATURES (VERY IMPORTANT!) ===
        # Living space ratios
        df['living_to_lot_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)
        df['living_to_lot15_ratio'] = df['sqft_living'] / (df['sqft_living15'] + 1)
        
        # Room density
        df['bedrooms_per_sqft'] = df['bedrooms'] / (df['sqft_living'] + 1) * 1000
        df['bathrooms_per_sqft'] = df['bathrooms'] / (df['sqft_living'] + 1) * 1000
        df['bedroom_to_bathroom_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.1)
        
        # Basement and above ground
        df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
        df['basement_ratio'] = df['sqft_basement'] / (df['sqft_living'] + 1)
        df['above_to_living_ratio'] = df['sqft_above'] / (df['sqft_living'] + 1)
        
        # Floor space distribution
        df['avg_sqft_per_floor'] = df['sqft_living'] / (df['floors'] + 0.5)
        
        print("✓ Created ratio features: 10 new ratio-based features")
        
        # === COMPOSITE FEATURES ===
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        df['quality_score'] = df['grade'] * df['condition']
        df['luxury_score'] = df['grade'] * df['view']
        
        # Overall attractiveness score
        df['attractiveness'] = (
            df['grade'] * 0.4 + 
            df['condition'] * 0.3 + 
            df['view'] * 0.2 + 
            df['waterfront'] * 10 * 0.1
        )
        
        print("✓ Created composite features: total_rooms, quality_score, luxury_score, attractiveness")
        
        # === LOCATION-BASED FEATURES ===
        # Average prices by zipcode
        df['zipcode_avg_price'] = df.groupby('zipcode')['price'].transform('mean')
        df['zipcode_median_price'] = df.groupby('zipcode')['price'].transform('median')
        df['price_per_sqft'] = df['price'] / (df['sqft_living'] + 1)
        df['zipcode_avg_price_per_sqft'] = df.groupby('zipcode')['price_per_sqft'].transform('mean')
        
        # Neighborhood comparison
        df['above_zipcode_avg'] = (df['price'] > df['zipcode_avg_price']).astype(int)
        
        # Location clusters using KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=10, random_state=self.random_state, n_init=10)
        df['location_cluster'] = kmeans.fit_predict(df[['lat', 'long']])
        
        # Distance from city center (approximate Seattle center)
        seattle_center = (47.6062, -122.3321)
        df['distance_from_center'] = np.sqrt(
            (df['lat'] - seattle_center[0])**2 + 
            (df['long'] - seattle_center[1])**2
        )
        
        print("✓ Created location features: zipcode stats, location_cluster, distance_from_center")
        
        # === CATEGORICAL ENCODING FEATURES ===
        # View quality categories
        df['has_good_view'] = (df['view'] >= 3).astype(int)
        
        # Condition categories
        df['is_excellent_condition'] = (df['condition'] >= 4).astype(int)
        
        # Grade categories
        df['grade_category'] = pd.cut(df['grade'], 
                                       bins=[0, 6, 9, 13], 
                                       labels=[0, 1, 2])  # Low, Medium, High
        
        print("✓ Created categorical encodings: has_good_view, is_excellent_condition, grade_category")
        
        # === DROP REDUNDANT FEATURES ===
        redundant_features = [
            'yr_built', 'yr_renovated',  # Already used for age features
            'sqft_above',  # Highly correlated with sqft_living
            'zipcode',  # Encoded via location features
            'price_per_sqft'  # Intermediate calculation
        ]
        
        df = df.drop(columns=[f for f in redundant_features if f in df.columns])
        print(f"\n✓ Dropped redundant features: {redundant_features}")
        
        print(f"\n✓ Total features after engineering: {len(df.columns) - 1}")  # -1 for price
        
        return df
    
    def handle_skewness(self, X, y):
        """Handle skewness in features and target."""
        print("\n" + "=" * 80)
        print("HANDLING SKEWNESS")
        print("=" * 80)
        
        # Check skewness for numeric features
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        skewness = X[numeric_cols].apply(lambda x: skew(x.dropna()))
        skewed_features = skewness[abs(skewness) > 0.75].index.tolist()
        
        print(f"\nFound {len(skewed_features)} skewed features (|skew| > 0.75)")
        
        # Log transform skewed features
        for feature in skewed_features:
            if (X[feature] >= 0).all():  # Only if all values are non-negative
                X[feature] = np.log1p(X[feature])
        
        print(f"✓ Applied log1p transformation to {len(skewed_features)} features")
        
        # Log transform target (house prices are typically right-skewed)
        y_log = np.log1p(y)
        print(f"✓ Applied log1p transformation to target variable")
        print(f"  Original price range: ${y.min():,.0f} - ${y.max():,.0f}")
        print(f"  Log-transformed range: {y_log.min():.2f} - {y_log.max():.2f}")
        
        return X, y_log
    
    def prepare_preprocessing_pipeline(self, X):
        """Create preprocessing pipeline."""
        print("\n" + "=" * 80)
        print("PREPARING PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # Define feature groups
        self.categorical_features = [
            'waterfront', 'view', 'condition', 'grade', 
            'location_cluster', 'season', 'is_renovated',
            'is_peak_season', 'is_new_house', 'has_basement',
            'has_good_view', 'is_excellent_condition', 'grade_category',
            'above_zipcode_avg', 'quarter_sold', 'month_sold'
        ]
        
        # Filter out features that might not exist
        self.categorical_features = [f for f in self.categorical_features if f in X.columns]
        
        # All other numeric features
        self.numeric_features = [col for col in X.columns 
                                if col not in self.categorical_features]
        
        print(f"\nNumeric features: {len(self.numeric_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        
        # Create preprocessor
        preprocessor = ColumnTransformer([
            ('num', RobustScaler(), self.numeric_features),
            ('cat', 'passthrough', self.categorical_features)
        ])
        
        self.feature_names = self.numeric_features + self.categorical_features
        
        print("✓ Preprocessing pipeline created")
        print("  Numeric: RobustScaler (handles outliers better)")
        print("  Categorical: Passthrough (already encoded)")
        
        return preprocessor
    
    def train_model(self, X_train, y_train, tune_hyperparameters=True):
        """Train model with optional hyperparameter tuning."""
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)
        
        # Create preprocessing pipeline
        preprocessor = self.prepare_preprocessing_pipeline(X_train)
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=self.random_state))
        ])
        
        if tune_hyperparameters:
            print("\n🔧 Tuning hyperparameters with RandomizedSearchCV...")
            print("This may take several minutes...\n")
            
            # Parameter distribution for randomized search
            param_distributions = {
                'regressor__n_estimators': randint(100, 600),
                'regressor__max_depth': [10, 20, 30, 40, None],
                'regressor__min_samples_split': randint(2, 20),
                'regressor__min_samples_leaf': randint(1, 10),
                'regressor__max_features': ['sqrt', 'log2', None],
                'regressor__bootstrap': [True, False]
            }
            
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions,
                n_iter=50,  # Number of parameter settings sampled
                cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
            
            random_search.fit(X_train, y_train)
            
            print(f"\n✓ Best cross-validation MAE: ${-random_search.best_score_:,.2f}")
            print(f"\n✓ Best parameters:")
            for param, value in random_search.best_params_.items():
                print(f"  {param}: {value}")
            
            self.model = random_search.best_estimator_
            
        else:
            print("\n🚀 Training with default parameters...")
            pipeline.fit(X_train, y_train)
            self.model = pipeline
            print("✓ Model trained")
        
        return self.model
    
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        """Comprehensive model evaluation."""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        # Make predictions (convert back from log scale)
        y_train_pred_log = self.model.predict(X_train)
        y_test_pred_log = self.model.predict(X_test)
        
        y_train_pred = np.expm1(y_train_pred_log)
        y_test_pred = np.expm1(y_test_pred_log)
        y_train_actual = np.expm1(y_train)
        y_test_actual = np.expm1(y_test)
        
        # Calculate metrics
        def calc_metrics(y_true, y_pred, set_name):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            print(f"\n📊 {set_name} Set Performance:")
            print(f"  MAE:  ${mae:,.2f}")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            
            return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}
        
        train_metrics = calc_metrics(y_train_actual, y_train_pred, "Train")
        test_metrics = calc_metrics(y_test_actual, y_test_pred, "Test")
        
        # Check for overfitting
        r2_gap = train_metrics['r2'] - test_metrics['r2']
        print(f"\n⚠️  Overfitting Check:")
        print(f"  R² gap (train - test): {r2_gap:.4f}")
        if r2_gap > 0.05:
            print(f"  ⚠️  Warning: Model may be overfitting (gap > 0.05)")
        else:
            print(f"  ✓ Model generalization looks good!")
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred,
                'train_actual': y_train_actual,
                'test_actual': y_test_actual
            }
        }
    
    def analyze_feature_importance(self, top_n=20):
        """Analyze and visualize feature importance."""
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        # Get feature importances
        importances = self.model.named_steps['regressor'].feature_importances_
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n📈 Top {top_n} Most Important Features:\n")
        print(feature_importance_df.head(top_n).to_string(index=False))
        
        # Visualize
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=feature_importance_df.head(top_n),
            x='importance',
            y='feature',
            palette='viridis'
        )
        plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✓ Feature importance plot saved")
        
        return feature_importance_df
    
    def error_analysis(self, X_test, evaluation_results):
        """Detailed error analysis."""
        print("\n" + "=" * 80)
        print("ERROR ANALYSIS")
        print("=" * 80)
        
        predictions = evaluation_results['predictions']
        y_test_actual = predictions['test_actual']
        y_test_pred = predictions['test']
        
        # Create error DataFrame
        error_df = X_test.copy()
        error_df['actual_price'] = y_test_actual
        error_df['predicted_price'] = y_test_pred
        error_df['error'] = error_df['actual_price'] - error_df['predicted_price']
        error_df['abs_error'] = abs(error_df['error'])
        error_df['error_pct'] = (error_df['abs_error'] / error_df['actual_price']) * 100
        
        # Price segments
        error_df['price_segment'] = pd.cut(
            error_df['actual_price'],
            bins=[0, 300000, 500000, 750000, np.inf],
            labels=['Low (<$300K)', 'Medium ($300K-$500K)', 
                   'High ($500K-$750K)', 'Luxury (>$750K)']
        )
        
        print("\n📊 Error Statistics by Price Segment:\n")
        segment_stats = error_df.groupby('price_segment').agg({
            'error_pct': ['mean', 'median', 'std'],
            'abs_error': ['mean', 'median']
        }).round(2)
        print(segment_stats)
        
        # Identify worst predictions
        print("\n\n⚠️  Top 5 Worst Predictions (by absolute error):\n")
        worst_predictions = error_df.nlargest(5, 'abs_error')[
            ['actual_price', 'predicted_price', 'abs_error', 'error_pct']
        ]
        print(worst_predictions.to_string(index=False))
        
        # Visualizations
        self._plot_error_analysis(error_df, predictions)
        
        return error_df
    
    def _plot_error_analysis(self, error_df, predictions):
        """Create comprehensive error analysis plots."""
        fig = plt.figure(figsize=(18, 12))
        
        y_test_actual = predictions['test_actual']
        y_test_pred = predictions['test']
        residuals = predictions['test_actual'] - predictions['test']
        
        # 1. Actual vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(y_test_actual, y_test_pred, alpha=0.4, s=20)
        ax1.plot([y_test_actual.min(), y_test_actual.max()],
                [y_test_actual.min(), y_test_actual.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Price ($)', fontsize=11)
        ax1.set_ylabel('Predicted Price ($)', fontsize=11)
        ax1.set_title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals Distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals ($)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals vs Predicted
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(y_test_pred, residuals, alpha=0.4, s=20)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted Price ($)', fontsize=11)
        ax3.set_ylabel('Residuals ($)', fontsize=11)
        ax3.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Error % by Price Segment
        ax4 = plt.subplot(2, 3, 4)
        error_df.boxplot(column='error_pct', by='price_segment', ax=ax4)
        ax4.set_xlabel('Price Segment', fontsize=11)
        ax4.set_ylabel('Error %', fontsize=11)
        ax4.set_title('Error Distribution by Price Segment', fontsize=12, fontweight='bold')
        plt.suptitle('')  # Remove default title
        ax4.grid(True, alpha=0.3)
        
        # 5. Q-Q Plot for Residuals
        ax5 = plt.subplot(2, 3, 5)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax5)
        ax5.set_title('Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Absolute Error Distribution
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(error_df['abs_error'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax6.set_xlabel('Absolute Error ($)', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('Distribution of Absolute Errors', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✓ Error analysis plots saved")
    
    def full_pipeline(self, filepath, tune_hyperparameters=True):
        """Run complete ML pipeline."""
        print("\n" + "=" * 80)
        print("🚀 HOUSE PRICE PREDICTION - COMPLETE PIPELINE")
        print("=" * 80)
        
        # 1. Load data
        df = self.load_data(filepath)
        
        # 2. Clean data
        df = self.clean_data(df)
        
        # 3. Engineer features
        df = self.engineer_features(df)
        
        # 4. Prepare X and y
        X = df.drop(columns=['price'])
        y = df['price']
        
        # 5. Handle skewness
        X, y_log = self.handle_skewness(X, y)
        
        # 6. Train-test split
        print("\n" + "=" * 80)
        print("TRAIN-TEST SPLIT")
        print("=" * 80)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=0.2, random_state=self.random_state
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")
        
        # 7. Train model
        self.train_model(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
        
        # 8. Evaluate model
        evaluation_results = self.evaluate_model(X_train, y_train, X_test, y_test)
        
        # 9. Feature importance
        feature_importance = self.analyze_feature_importance(top_n=20)
        
        # 10. Error analysis
        error_analysis = self.error_analysis(X_test, evaluation_results)
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📁 Generated files:")
        print("  - feature_importance.png")
        print("  - error_analysis.png")
        print("\n💡 Model is ready for predictions!")
        
        return {
            'model': self.model,
            'evaluation': evaluation_results,
            'feature_importance': feature_importance,
            'error_analysis': error_analysis
        }


# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == "__main__":
    # Initialize predictor
    predictor = HousePricePredictor(random_state=42)
    
    # Run full pipeline
    results = predictor.full_pipeline(
        filepath="kc_house_data.csv",
        tune_hyperparameters=True  # Set to False for quick testing
    )
    
    print("\n" + "=" * 80)
    print("🎉 ALL DONE! Check the outputs folder for visualizations.")
    print("=" * 80)
