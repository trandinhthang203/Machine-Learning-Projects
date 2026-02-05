from projects.src.components.data_ingestion import DataIngestion
from projects.src.components.data_validation import DataValidation
from projects.src.components.data_transformation import DataTransformation
from projects.src.components.model_trainer import ModelTrainer
from projects.src.components.model_evaluation import ModelEvaluation
import os
import sys
from utils.logger import logging
from utils.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()
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