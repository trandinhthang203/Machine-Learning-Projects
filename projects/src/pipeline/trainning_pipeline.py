# projects/src/pipeline/training_pipeline.py
"""
Main training pipeline orchestrating toàn bộ quá trình training
"""
import sys
from projects.src.utils.logger import logging
from projects.src.utils.exception import CustomException
from projects.src.components.data_ingestion import DataIngestion
from projects.src.components.data_validation import DataValidation
from projects.src.components.data_transformation import DataTransformation
from projects.src.components.model_trainer import ModelTrainer
from projects.src.components.model_evaluation import ModelEvaluation
from projects.src.config.configuration import ConfiguartionManager
from datetime import datetime
import json
import os


class TrainingPipeline:
    """
    End-to-end training pipeline
    Orchestrates: Data Ingestion -> Validation -> Transformation -> Training -> Evaluation
    """
    
    def __init__(self):
        self.pipeline_name = "Training Pipeline"
        self.start_time = None
        self.end_time = None
        self.status = "initialized"
        self.results = {}
        
    def log_pipeline_start(self):
        """Log pipeline start"""
        self.start_time = datetime.now()
        logging.info(f"{'='*60}")
        logging.info(f"{self.pipeline_name} STARTED")
        logging.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"{'='*60}")
    
    def log_pipeline_end(self):
        """Log pipeline end và summary"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        logging.info(f"{'='*60}")
        logging.info(f"{self.pipeline_name} COMPLETED")
        logging.info(f"End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logging.info(f"Status: {self.status}")
        logging.info(f"{'='*60}")
        
        # Save pipeline results
        self._save_pipeline_results(duration)
    
    def _save_pipeline_results(self, duration):
        """Save pipeline execution results"""
        pipeline_results = {
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": duration,
            "status": self.status,
            "results": self.results
        }
        
        # Save to artifacts
        results_dir = "projects/artifacts/pipeline_results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(
            results_dir,
            f"training_pipeline_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=4)
        
        logging.info(f"Pipeline results saved to: {results_file}")
    
    def run_data_ingestion(self):
        """Step 1: Data Ingestion"""
        try:
            logging.info("\n" + "="*50)
            logging.info("STEP 1: DATA INGESTION")
            logging.info("="*50)
            
            data_ingestion = DataIngestion()
            data_ingestion.init_data_ingestion()
            
            self.results['data_ingestion'] = {
                'status': 'success',
                'message': 'Data ingested successfully'
            }
            
            logging.info("✓ Data Ingestion Completed")
            return True
            
        except Exception as e:
            self.results['data_ingestion'] = {
                'status': 'failed',
                'error': str(e)
            }
            logging.error(f"✗ Data Ingestion Failed: {str(e)}")
            raise CustomException(e, sys)
    
    def run_data_validation(self):
        """Step 2: Data Validation"""
        try:
            logging.info("\n" + "="*50)
            logging.info("STEP 2: DATA VALIDATION")
            logging.info("="*50)
            
            data_validation = DataValidation()
            validation_status = data_validation.init_data_validation()
            
            self.results['data_validation'] = {
                'status': 'success',
                'validation_passed': validation_status,
                'message': 'Data validation completed'
            }
            
            if not validation_status:
                logging.warning("⚠ Data Validation: Some checks failed")
            else:
                logging.info("✓ Data Validation Completed")
            
            return validation_status
            
        except Exception as e:
            self.results['data_validation'] = {
                'status': 'failed',
                'error': str(e)
            }
            logging.error(f"✗ Data Validation Failed: {str(e)}")
            raise CustomException(e, sys)
    
    def run_data_transformation(self):
        """Step 3: Data Transformation"""
        try:
            logging.info("\n" + "="*50)
            logging.info("STEP 3: DATA TRANSFORMATION")
            logging.info("="*50)
            
            data_transformation = DataTransformation()
            train_path, test_path, preprocessor_path = \
                data_transformation.init_data_transformation()
            
            self.results['data_transformation'] = {
                'status': 'success',
                'train_data_path': train_path,
                'test_data_path': test_path,
                'preprocessor_path': preprocessor_path,
                'message': 'Data transformed successfully'
            }
            
            logging.info("✓ Data Transformation Completed")
            return train_path, test_path, preprocessor_path
            
        except Exception as e:
            self.results['data_transformation'] = {
                'status': 'failed',
                'error': str(e)
            }
            logging.error(f"✗ Data Transformation Failed: {str(e)}")
            raise CustomException(e, sys)
    
    def run_model_training(self):
        """Step 4: Model Training"""
        try:
            logging.info("\n" + "="*50)
            logging.info("STEP 4: MODEL TRAINING")
            logging.info("="*50)
            
            model_trainer = ModelTrainer()
            model_path = model_trainer.init_model_traner()
            
            self.results['model_training'] = {
                'status': 'success',
                'model_path': model_path,
                'message': 'Model trained successfully'
            }
            
            logging.info("✓ Model Training Completed")
            return model_path
            
        except Exception as e:
            self.results['model_training'] = {
                'status': 'failed',
                'error': str(e)
            }
            logging.error(f"✗ Model Training Failed: {str(e)}")
            raise CustomException(e, sys)
    
    def run_model_evaluation(self):
        """Step 5: Model Evaluation"""
        try:
            logging.info("\n" + "="*50)
            logging.info("STEP 5: MODEL EVALUATION")
            logging.info("="*50)
            
            model_evaluation = ModelEvaluation()
            metrics = model_evaluation.init_model_evaluation()
            
            self.results['model_evaluation'] = {
                'status': 'success',
                'metrics': metrics,
                'message': 'Model evaluated successfully'
            }
            
            logging.info("✓ Model Evaluation Completed")
            return metrics
            
        except Exception as e:
            self.results['model_evaluation'] = {
                'status': 'failed',
                'error': str(e)
            }
            logging.error(f"✗ Model Evaluation Failed: {str(e)}")
            raise CustomException(e, sys)
    
    def run_pipeline(self, skip_validation=False):
        """
        Execute complete training pipeline
        
        Args:
            skip_validation (bool): Skip data validation step if True
        
        Returns:
            dict: Pipeline results
        """
        try:
            self.log_pipeline_start()
            
            # Step 1: Data Ingestion
            self.run_data_ingestion()
            
            # Step 2: Data Validation (optional)
            if not skip_validation:
                validation_status = self.run_data_validation()
                if not validation_status:
                    logging.warning("Data validation failed but continuing pipeline...")
            
            # Step 3: Data Transformation
            train_path, test_path, preprocessor_path = self.run_data_transformation()
            
            # Step 4: Model Training
            model_path = self.run_model_training()
            
            # Step 5: Model Evaluation
            metrics = self.run_model_evaluation()
            
            self.status = "completed"
            self.log_pipeline_end()
            
            return {
                'status': 'success',
                'message': 'Training pipeline completed successfully',
                'results': self.results
            }
            
        except Exception as e:
            self.status = "failed"
            self.log_pipeline_end()
            logging.error(f"Pipeline failed with error: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        results = pipeline.run_pipeline()
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)
