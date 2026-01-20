from src.entity.config_entity import DataValidationConfig
from src.exception import CustomException
import sys

class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def init_data_validation_config(self):
        try:
            # VALIDATE COLUMNS

            #VALIDATE TYPE COLUMNS

            pass
        except Exception as e:
            raise CustomException(e, sys)