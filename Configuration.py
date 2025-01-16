import yaml
from src.entity.config_entity import DataIngestionConfig

class Configuration:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as file:
            self.config_data = yaml.load(file, Loader=yaml.FullLoader)
    
    def get_data_ingestion_config(self):
        data_ingestion_config = self.config_data['data_ingestion']
        return DataIngestionConfig(data_ingestion_config['raw_data_dir'], data_ingestion_config['file_name'])






import yaml
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    OutputConfig,
    AdditionalConfig,
)


class Configuration:
    def __init__(self, config_file: str):
        with open(config_file, "r") as file:
            self.config_data = yaml.load(file, Loader=yaml.FullLoader)

    def get_data_ingestion_config(self):
        """Retrieve data ingestion configuration."""
        data_ingestion_config = self.config_data["data_ingestion"]
        return DataIngestionConfig(
            raw_data_dir=data_ingestion_config["raw_data_dir"],
            file_name=data_ingestion_config["file_name"],
        )

    def get_data_transformation_config(self):
        """Retrieve data transformation configuration."""
        data_transformation_config = self.config_data["data_transformation"]
        return DataTransformationConfig(
            transformation_method=data_transformation_config["transformation_method"]
        )

    def get_model_training_config(self):
        """Retrieve model training configuration."""
        model_training_config = self.config_data["model_training"]
        return ModelTrainingConfig(
            test_size=model_training_config["test_size"],
            random_state=model_training_config["random_state"],
            model_type=model_training_config["model_type"],
            n_estimators=model_training_config["n_estimators"],
            max_depth=model_training_config["max_depth"],
        )

    def get_model_evaluation_config(self):
        """Retrieve model evaluation configuration."""
        model_evaluation_config = self.config_data["model_evaluation"]
        return ModelEvaluationConfig(
            metrics=model_evaluation_config["metrics"],
            threshold=model_evaluation_config["threshold"],
        )

    def get_output_config(self):
        """Retrieve output paths configuration."""
        output_config = self.config_data["output"]
        return OutputConfig(
            model_save_path=output_config["model_save_path"],
            logs_path=output_config["logs_path"],
            model_file_name=output_config["model_file_name"],
        )

    def get_additional_config(self):
        """Retrieve additional configuration."""
        additional_config = self.config_data["additional"]
        return AdditionalConfig(
            logging_level=additional_config["logging_level"],
            enable_logging=additional_config["enable_logging"],
        )
        
