import os
import sys
from projects.src.utils.logger import logging
from projects.src.utils.exception import CustomException
from src.entity.config_entity import DataIngestionConfig
from src.config.configuration import ConfiguartionManager
from urllib.request import urlretrieve
import zipfile

class DataIngestion:
    '''
        Download / collect data
        Lưu raw data
        (Có thể) unzip
    '''
    def __init__(self):
        data_ingestion = ConfiguartionManager()
        self.ingestion_config = data_ingestion.get_data_ingestion_config()


    def init_data_ingestion(self):
        logging.info("Data ingestion starting...")

        try:
            # download file
            if not os.path.exists(self.ingestion_config.local_data_file):
                url = self.ingestion_config.source_url
                filename = self.ingestion_config.local_data_file

                path, headers =  urlretrieve(url, filename)
                logging.info(f"{path} download with info {headers}")
            else:
                logging.info(f"{self.ingestion_config.local_data_file} already exist.")

            # extract file
            if not os.path.exists(self.ingestion_config.unzip_dir):
                os.makedirs(self.ingestion_config.unzip_dir)
                logging.info(f"Created directory {self.ingestion_config.unzip_dir}")

            if zipfile.is_zipfile(self.ingestion_config.local_data_file):
                with zipfile.ZipFile(self.ingestion_config.local_data_file, "r") as zip:
                    zip.extractall(self.ingestion_config.unzip_dir)
            else:
                csv_path = self.ingestion_config.local_data_file.replace(".zip", ".csv")
                # syntax: os.rename(src, dst)
                # src: current path
                # dst: replace path
                if not os.path.exists(csv_path):
                    os.rename(self.ingestion_config.local_data_file, csv_path)
                    logging.info("Downloaded file is CSV, skipping unzip step.")
                else:
                    logging.info("File CSV already exist.")

        except Exception as e:
            raise CustomException(e, sys)
        