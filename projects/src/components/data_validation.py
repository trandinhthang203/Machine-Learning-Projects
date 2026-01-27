from projects.src.entity.config_entity import DataValidationConfig
from projects.src.config.configuration import ConfiguartionManager
from projects.src.utils.exception import CustomException
import sys
import pandas as pd
from projects.src.utils.logger import logging
from datetime import datetime

class DataValidation:
    '''
        _validate_columns
        _validate_data_types
        _validate_missing_values
        _validate_duplicates
    '''
    def __init__(self):
        data_validation = ConfiguartionManager()
        self.validation_config = data_validation.get_data_validation_config()
        self.validation_report = {
            "timestamp": datetime.now().isoformat(),
            "validations": {},
            "overall_status": True
        }


    def _validate_columns(self) -> bool:
        logging.info("Validating column names...")

        schemas_cols = set(self.validation_config.all_schema)
        df_cols = set((pd.read_csv(self.validation_config.unzip_dir_data)).columns)

        missing_cols = schemas_cols - df_cols
        extra_cols = df_cols - schemas_cols

        status = len(missing_cols) == 0 and len(extra_cols) == 0

        self.validation_report["validations"]["columns"] = {
            "status": status,
            "expected_columns": schemas_cols,
            "actual_columns": df_cols,
            "missing columns": missing_cols,
            "extra columns": extra_cols
        }

        if missing_cols:
            logging.info(f"Missing columns: {missing_cols}")
        if extra_cols:
            logging.info(f"Extra columns: {extra_cols}")\
            
        logging.info("Validation column names done.")

        return status

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
        

if __name__ == "__main__":
    val = DataValidation()
    val._validate_columns()