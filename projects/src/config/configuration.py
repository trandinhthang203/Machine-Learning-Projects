from projects.src.constants.constant import *
from utils.common import read_yaml, create_dir
from src.entity.config_entity import DataIngestionConfig


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

