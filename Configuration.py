import yaml
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:
    def __init__(self, config_file_path: str = "config/config.yaml"):
        self.config_file_path = config_file_path
        self.config_data = self.read_yaml_file()

    def read_yaml_file(self):
        """Reads the YAML configuration file."""
        try:
            with open(self.config_file_path, "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Error reading config file: {e}")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        ingestion_config = self.config_data["data_ingestion"]
        return DataIngestionConfig(
            input_file_path=ingestion_config["input_file_path"],
            output_train_path=ingestion_config["output_train_path"],
            output_test_path=ingestion_config["output_test_path"],
            test_size=ingestion_config["test_size"],
            random_state=ingestion_config["random_state"],
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        validation_config = self.config_data["data_validation"]
        return DataValidationConfig(
            schema_file_path=validation_config["schema_file_path"],
            required_columns=validation_config["required_columns"],
            target_column=validation_config["target_column"],
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        transformation_config = self.config_data["data_transformation"]
        return DataTransformationConfig(
            tfidf_max_features=transformation_config["tfidf_max_features"],
            tfidf_min_df=transformation_config["tfidf_min_df"],
            tfidf_max_df=transformation_config["tfidf_max_df"],
            target_column=transformation_config["target_column"],
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        trainer_config = self.config_data["model_trainer"]
        return ModelTrainerConfig(
            model_type=trainer_config["model_type"],
            n_estimators=trainer_config["n_estimators"],
            random_state=trainer_config["random_state"],
            model_save_path=trainer_config["model_save_path"],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        evaluation_config = self.config_data["model_evaluation"]
        return ModelEvaluationConfig(
            metric_name=evaluation_config["metric_name"],
            metric_threshold=evaluation_config["metric_threshold"],
            evaluation_report_path=evaluation_config["evaluation_report_path"],
        )
      
