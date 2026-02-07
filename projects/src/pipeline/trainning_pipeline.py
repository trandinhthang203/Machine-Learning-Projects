from projects.src.components.data_ingestion import DataIngestion
from projects.src.components.data_validation import DataValidation
from projects.src.components.data_transformation import DataTransformation
from projects.src.components.model_trainer import ModelTrainer
from projects.src.components.model_evaluation import ModelEvaluation
from projects.src.config.configuration import ConfiguartionManager
from datetime import datetime
import os
import sys
from projects.src.utils.logger import logging
from projects.src.utils.exception import CustomException
import json

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()
        self.config = ConfiguartionManager()
        self.results = {}

    def run_data_ingestion(self):
        try:
            logging.info("STEP 1: DATA INGESTION")
            self.data_ingestion.init_data_ingestion()

            self.results["data_ingestion"] = {
                "status": "success",
                "message": "Data ingestion successfully"
            }

        except Exception as e:
            self.results["data_ingestion"] = {
                "status": "failed",
                "error": str(e)
            }
            raise CustomException(e, sys)
        
    def run_data_validation(self):
        try:
            logging.info("STEP 2: DATA VALIDATION")
            overall_status = self.data_validation.init_data_validation()

            self.results["data_validation"] = {
                "status": "success",
                "validation_passed": overall_status["overall_status"],
                "message": "Data validation successfully"
            }

            if overall_status["overall_status"]:
                logging.info("Data Validation Completed")
            else:
                logging.warning("Data Validation: Some checks failed")
                
            return overall_status["overall_status"]

        except Exception as e:
            self.results["data_validation"] = {
                "status": "failed",
                "error": str(e)
            }
            raise CustomException(e, sys)

    def run_data_transformation(self):
        try:
            logging.info("STEP 3: DATA TRANSFORMATION")
            self.data_transformation.init_data_transformation()

            self.results["data_transformation"] = {
                "status": "success",
                "message": "Data transformation completed"
            }

        except Exception as e:
            self.results["data_transformation"] = {
                "status": "failed",
                "error": str(e)
            }
            raise CustomException(e, sys)
        
    def run_model_trainer(self):
        try:
            logging.info("STEP 4: MODEL TRAINER")
            model_path = self.model_trainer.init_model_traner()

            self.results["model_trainer"] = {
                "status": "success",
                "model_path": model_path,
                "message": "Model trained successfully"
            }

        except Exception as e:
            self.results["model_trainer"] = {
                "status": "failed",
                "error": str(e)
            }
            raise CustomException(e, sys)
        
    def run_model_evaluation(self):
        try:
            logging.info("STEP 5: MODEL EVALUATION")
            reports_path = self.model_evaluation.init_model_evaluation()

            self.results["model_evaluation"] = {
                "status": "success",
                "report_path": reports_path,
                "message": "Model evaluated successfully"
            }

        except Exception as e:
            self.results["model_evaluation"] = {
                "status": "failed",
                "error": str(e)
            }
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        try:
            # self.run_data_ingestion()
            status = self.run_data_validation()
            if status:
                self.run_data_transformation()
                self.run_model_trainer()
                self.run_model_evaluation()

            config = self.config.get_training_pipeline_config()
            with open(os.path.join(config.root_dir,  f"training_pipeline_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json"), "w") as file:
                json.dump(self.results, file, indent=4)
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()