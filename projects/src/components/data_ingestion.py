import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def init_data_ingestion(self):
        logging.info("Data ingestion starting...")

        try:
            # download file

            # extract file


            pass
        except Exception as e:
            raise CustomException(e, sys)
        