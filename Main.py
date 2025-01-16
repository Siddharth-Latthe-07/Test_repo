from src.config.configuration import ConfigurationManager
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_validation_pipeline import DataValidationPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.pipeline.model_eval_pipeline import ModelEvalPipeline


def main():
    # Load configuration from YAML file
    config_manager = ConfigurationManager(config_file_path="config/config.yaml")
    
    # Data Ingestion
    print("Starting Data Ingestion...")
    data_ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion_pipeline = DataIngestionPipeline(config=data_ingestion_config)
    data_ingestion_pipeline.run()

    # Data Validation
    print("Starting Data Validation...")
    data_validation_config = config_manager.get_data_validation_config()
    data_validation_pipeline = DataValidationPipeline(config=data_validation_config)
    data_validation_pipeline.run()

    # Data Transformation
    print("Starting Data Transformation...")
    data_transformation_config = config_manager.get_data_transformation_config()
    data_transformation_pipeline = DataTransformationPipeline(config=data_transformation_config)
    data_transformation_pipeline.run()

    # Model Training
    print("Starting Model Training...")
    model_trainer_config = config_manager.get_model_trainer_config()
    model_trainer_pipeline = ModelTrainerPipeline(config=model_trainer_config)
    model_trainer_pipeline.run()

    # Model Evaluation
    print("Starting Model Evaluation...")
    model_evaluation_config = config_manager.get_model_evaluation_config()
    model_eval_pipeline = ModelEvalPipeline(config=model_evaluation_config)
    model_eval_pipeline.run()

    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
  
