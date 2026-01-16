import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts", "raw.csv")
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("Data ingestion starting...")

        try:
            df = pd.read_csv(r"D:\MLs\ML\kc_house_data.csv")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

            df_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    instance = DataIngestion()
    instance.init_data_ingestion()