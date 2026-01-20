from src.constants.constant import *
from src.utils.common import read_yaml, create_dir
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig


class ConfiguartionManager:
    def __init__(self):
        self.config_file_path = read_yaml(CONFIG_FILE_PATH)
        self.params_file_path = read_yaml(PARAMS_FILE_PATH)
        self.schema_file_path = read_yaml(SCHEMA_FILE_PATH)

        create_dir([self.config_file_path.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config_file_path.data_ingestion
        create_dir([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config_file_path.data_validation
        schema = self.schema_file_path.COLUMNS
        create_dir([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            unzip_dir_data=config.unzip_dir_data,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema
        )
        return data_validation_config

