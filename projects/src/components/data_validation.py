from src.entity.config_entity import DataValidationConfig
from src.config.configuration import ConfiguartionManager
from projects.src.utils.exception import CustomException
import sys
import pandas as pd
from projects.src.utils.logger import logging

class DataValidation:
    '''
        Kiểm tra schema
        Kiểm tra missing values
        Kiểm tra data drift
        Validate columns
    '''
    def __init__(self):
        data_validation = ConfiguartionManager()
        self.validation_config = data_validation.get_data_validation_config()

    def init_data_validation(self):
        try:
            # VALIDATE COLUMNS
            df = pd.read_csv(self.validation_config.unzip_dir_data)
            schema = self.validation_config.all_schema
            status_file = True

            for column in df.columns:
                if column not in schema.keys():
                    status_file = False
                    break

            with open(self.validation_config.STATUS_FILE, "w") as file:
                file.write(str(status_file))
                logging.info(str(status_file))

            # VALIDATE TYPE COLUMNS
            

        except Exception as e:
            raise CustomException(e, sys)
        