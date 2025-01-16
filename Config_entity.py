class DataIngestionConfig:
    def __init__(self, raw_data_dir: str, file_name: str):
        self.raw_data_dir = raw_data_dir
        self.file_name = file_name





from dataclasses import dataclass
from typing import List


@dataclass
class DataIngestionConfig:
    raw_data_dir: str
    file_name: str


@dataclass
class DataTransformationConfig:
    transformation_method: str


@dataclass
class ModelTrainingConfig:
    test_size: float
    random_state: int
    model_type: str
    n_estimators: int
    max_depth: int


@dataclass
class ModelEvaluationConfig:
    metrics: List[str]
    threshold: float


@dataclass
class OutputConfig:
    model_save_path: str
    logs_path: str
    model_file_name: str


@dataclass
class AdditionalConfig:
    logging_level: str
    enable_logging: bool
    
        
