from projects.src.entity.config_entity import DataValidationConfig
from projects.src.config.configuration import ConfiguartionManager
from projects.src.utils.exception import CustomException
from projects.src.utils.common import write_json
import sys
import pandas as pd
from projects.src.utils.logger import logging
from datetime import datetime
from pathlib import Path
import os


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


    def _validate_columns(self, df: pd.DataFrame) -> bool:
        logging.info("Validating column names...")

        schemas_cols = set(self.validation_config.all_schema)
        df_cols = set(df.columns)

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
            logging.info(f"Extra columns: {extra_cols}")
            
        logging.info("Validation column names done.")

        return status


    def _validate_data_types(self, df: pd.DataFrame) -> bool:
        logging.info("Validating data types...")

        schemas = self.validation_config.all_schema
        df_cols = df.columns
        df_dict = {col: str(df[col].dtype) for col in list(df_cols)}
        
        type_mismatches = []
        for col, type in schemas.items():
            if type != df_dict[col]:
                type_mismatches.append({
                    "column": col,
                    "expected": type,
                    "actual": df_dict[col]
                })

        status = len(type_mismatches) == 0
        
        if type_mismatches:
            logging.info(f"Type mismatches: {type_mismatches}")
            
        logging.info("Validation data types done.")

        self.validation_report["validations"]["data_types"] = {
            "status": status,
            "type mismatches": type_mismatches
        }

        return status

    def _validate_missing_values(self, df: pd.DataFrame):
        logging.info("Validating missing value...")

        miss_cols = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count:
                miss_cols.append({
                    "missing column": col,
                    "missing count": int(missing_count)
                })

        status = len(miss_cols) == 0
        
        if miss_cols:
            logging.info(f"Columns with missing: {miss_cols}")
            
        logging.info("Validation missing value done.")

        self.validation_report["validations"]["missing_values"] = {
            "status": status,
            "coulmns_with_missing": miss_cols
        }

        return status

    def _validate_duplicates(self, df: pd.DataFrame):
        logging.info("Validating duplicates...")

        n_duplicates = int(df.duplicated().sum())
        status = n_duplicates == 0

        self.validation_report["validations"]["duplicates"] = {
            "status": status,
            "dupliactes count": n_duplicates
        }
        if n_duplicates > 0:
            logging.warning(f"Found {n_duplicates} duplicate rows")

        return status
    
    def _save_validation_report(self):

        report_path = os.path.join(self.validation_config.root_dir, self.validation_config.validation_report_name)
        write_json(Path(report_path), self.validation_report)
        logging.info(f"Validation report saved: {report_path}")
                     
    def init_data_validation(self):
        try:
            df = pd.read_csv(self.validation_config.unzip_dir_data)
            
            validations = [
                self._validate_columns(df),
                self._validate_data_types(df),
                self._validate_missing_values(df),
                self._validate_duplicates(df)
            ]

            self.validation_report["overall_status"] = all(validations)
            self._save_validation_report()

            overall_status = {
                "overall_status": self.validation_report["overall_status"]
            }
            write_json(self.validation_config.STATUS_FILE, overall_status)

            return overall_status

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    val = DataValidation()
    val.init_data_validation()
    # val._validate_columns()