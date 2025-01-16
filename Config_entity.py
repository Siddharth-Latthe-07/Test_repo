from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionConfig:
    input_file_path: str
    output_train_path: str
    output_test_path: str
    test_size: float
    random_state: int

@dataclass
class DataValidationConfig:
    schema_file_path: str
    required_columns: list
    target_column: str

@dataclass
class DataTransformationConfig:
    tfidf_max_features: Optional[int]
    tfidf_min_df: int
    tfidf_max_df: float
    target_column: str

@dataclass
class ModelTrainerConfig:
    model_type: str  # e.g., "RandomForestClassifier"
    n_estimators: int
    random_state: int
    model_save_path: str

@dataclass
class ModelEvaluationConfig:
    metric_name: str  # e.g., "accuracy"
    metric_threshold: Optional[float]
    evaluation_report_path: str
    
