from src.config.configuration import Configuration
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_validation_pipeline import DataValidationPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.pipeline.model_eval_pipeline import ModelEvalPipeline
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    # Load the configuration
    config = Configuration(config_file="config/config.yaml")

    # Fetch configurations
    data_ingestion_config = config.get_data_ingestion_config()
    data_transformation_config = config.get_data_transformation_config()
    model_training_config = config.get_model_training_config()
    model_evaluation_config = config.get_model_evaluation_config()
    output_config = config.get_output_config()

    # Configure MLflow
    mlflow.set_tracking_uri(output_config.logs_path)
    experiment_name = "Tense Classification Pipeline"
    mlflow.set_experiment(experiment_name)

    # Start MLflow Run
    with mlflow.start_run():
        try:
            # Log configuration parameters
            mlflow.log_param("data_ingestion_config", data_ingestion_config)
            mlflow.log_param("data_transformation_config", data_transformation_config)
            mlflow.log_param("model_training_config", model_training_config)
            mlflow.log_param("model_evaluation_config", model_evaluation_config)

            # Data Ingestion
            print("Starting Data Ingestion...")
            data_ingestion_pipeline = DataIngestionPipeline(config=data_ingestion_config)
            data = data_ingestion_pipeline.run()
            print("Data Ingestion Completed!")

            # Data Validation
            print("Starting Data Validation...")
            data_validation_pipeline = DataValidationPipeline(data=data)
            validated_data = data_validation_pipeline.run()
            print("Data Validation Completed!")

            # Data Transformation
            print("Starting Data Transformation...")
            data_transformation_pipeline = DataTransformationPipeline(data=validated_data)
            X, y, vectorizer = data_transformation_pipeline.run()
            print("Data Transformation Completed!")

            # Model Training
            print("Starting Model Training...")
            model_trainer_pipeline = ModelTrainerPipeline(X, y, model_training_config, output_config)
            model = model_trainer_pipeline.run()
            print("Model Training Completed!")

            # Model Evaluation
            print("Starting Model Evaluation...")
            model_eval_pipeline = ModelEvalPipeline(model, X, y, model_evaluation_config)
            model_eval_pipeline.run()
            print("Model Evaluation Completed!")

            print("Pipeline executed successfully!")

        except Exception as e:
            print(f"An error occurred: {e}")
            mlflow.log_param("pipeline_status", "Failed")
            mlflow.log_param("error_message", str(e))
            raise e

        finally:
            mlflow.log_param("pipeline_status", "Completed")
            
