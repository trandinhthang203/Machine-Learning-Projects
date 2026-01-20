from src.entity.config_entity import DataValidationConfig
from src.config.configuration import ConfiguartionManager
from src.exception import CustomException
import sys
import pandas as pd
from src.logger import logging

class DataValidation:
    def __init__(self):
        data_validation = ConfiguartionManager()
        self.validation_config = data_validation.get_data_validation_config()

    def init_data_validation(self):
        try:
            # VALIDATE COLUMNS
            df = pd.read_csv(self.validation_config.unzip_dir_data)
            columns = df.columns

            for column in columns:
                logging.info(column)

            #VALIDATE TYPE COLUMNS

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    ingestion = DataValidation()
    ingestion.init_data_validation()