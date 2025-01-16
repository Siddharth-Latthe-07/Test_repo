from src.config.configuration import Configuration
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_validation_pipeline import DataValidationPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.pipeline.model_eval_pipeline import ModelEvalPipeline

if __name__ == "__main__":
    # Load the config
    config = Configuration(config_file="config/config.yaml")

    # Data Ingestion
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion_pipeline = DataIngestionPipeline(config=data_ingestion_config)
    data = data_ingestion_pipeline.run()

    # Data Validation
    data_validation_pipeline = DataValidationPipeline(data=data)
    validated_data = data_validation_pipeline.run()

    # Data Transformation
    data_transformation_pipeline = DataTransformationPipeline(data=validated_data)
    X, y, vectorizer = data_transformation_pipeline.run()

    # Model Training
    model_trainer_pipeline = ModelTrainerPipeline(X, y)
    model = model_trainer_pipeline.run()

    # Model Evaluation
    model_eval_pipeline = ModelEvalPipeline(model, X, y)
    model_eval_pipeline.run()
    
