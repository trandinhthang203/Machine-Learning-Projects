from src.constants.constant import *
from src.utils.common import read_yaml, create_dir
from src.entity.config_entity import (
    DataIngestionConfig, 
    DataValidationConfig, 
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)


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
        schema = self.schema_file_path.columns
        create_dir([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            unzip_dir_data=config.unzip_dir_data,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema
        )
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config_file_path.data_transformation
        create_dir([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config_file_path.model_trainer
        params = self.params_file_path.ElasticNet
        target_column = self.schema_file_path.target_column
        create_dir([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_path=config.train.path,
            test_path=config.test_path,
            model_name=config.model_name,
            alpha=params.alpha,
            l1_ratio=params.l1_ratio,
            target_column=target_column
        )
        return model_trainer_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config_file_path.model_evaluation
        params: dict = self.params_file_path.ElasticNet
        target_column = self.schema_file_path.target_column
        create_dir([[config.root_dir]])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_path=config.test_path,
            model_path=config.model_path,
            params=params,
            metric_file_name=config.metric_file_name,
            target_column=target_column
        )
        return model_evaluation_config