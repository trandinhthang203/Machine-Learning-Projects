from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_dir_data: Path
    STATUS_FILE: str
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    data_transform_path: Path
    num_columns: List[str]
    cat_columns: List[str]


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    train_path: Path
    test_path: Path
    model_name: str
    alpha: float
    l1_ratio: float
    test_size: float
    target_column: str
    num_columns: list[str]
    cat_columns: list[str]


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_path: Path
    model_path: Path
    params: dict
    metric_file_name: float
    target_column: str
