from pathlib import Path
from box import ConfigBox
import yaml
import os
from src.constants.constant import *
from src.logger import logging
from typing import List

def read_yaml(path: Path) -> ConfigBox:
    with open(path, 'r') as file:
        content = yaml.safe_load(file)
        content = ConfigBox(content)
    return content


def create_dir(dir_list: List):
    for path in dir_list:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created directoriy at {path}")

if __name__ == "__main__":
    cont = create_dir(["src/test/hi.py"])
