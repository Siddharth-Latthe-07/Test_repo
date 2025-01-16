import mlflow
import mlflow.sklearn
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_validation_pipeline import DataValidationPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline
from src.config.configuration import ConfigurationManager

def main():
    # Initialize configuration manager
    config = ConfigurationManager()

    # Set up MLflow experiment
    experiment_name = "MLflow Pipeline Experiment"
    mlflow.set_tracking_uri("file:///mlruns")  # Local directory for MLflow tracking
    mlflow.set_experiment(experiment_name)

    # Start an MLflow run
    with mlflow.start_run(run_name="Pipeline Execution") as run:
        try:
            # Data Ingestion
            ingestion_pipeline = DataIngestionPipeline(config)
            raw_data = ingestion_pipeline.run()
            mlflow.log_param("raw_data_path", config.data_ingestion.raw_data_dir)

            # Data Validation
            validation_pipeline = DataValidationPipeline(config)
            validation_status = validation_pipeline.run(raw_data)
            mlflow.log_metric("data_validation_status", int(validation_status))

            # Data Transformation
            transformation_pipeline = DataTransformationPipeline(config)
            transformed_data, vectorizer = transformation_pipeline.run(raw_data)
            mlflow.log_param("transformation_method", config.data_transformation.transformation_method)

            # Save vectorizer as an artifact
            vectorizer_path = "artifacts/vectorizer.pkl"
            transformation_pipeline.save_vectorizer(vectorizer_path)
            mlflow.log_artifact(vectorizer_path)

            # Model Training
            trainer_pipeline = ModelTrainerPipeline(config)
            model, train_test_split_details = trainer_pipeline.run(transformed_data)
            mlflow.log_param("train_test_ratio", train_test_split_details["train_test_ratio"])
            mlflow.log_param("model_type", config.model_training.model_type)
            mlflow.log_param("n_estimators", config.model_training.n_estimators)
            mlflow.log_param("max_depth", config.model_training.max_depth)

            # Save model as an artifact
            model_path = "artifacts/model.pkl"
            trainer_pipeline.save_model(model_path)
            mlflow.log_artifact(model_path)

            # Model Evaluation
            evaluation_pipeline = ModelEvaluationPipeline(config)
            metrics = evaluation_pipeline.run(model, transformed_data)
            mlflow.log_metric("accuracy", metrics["accuracy"])

            # Log additional evaluation metrics
            for key, value in metrics.items():
                if key != "accuracy":
                    mlflow.log_metric(key, value)

            print("MLflow pipeline execution completed successfully.")

        except Exception as e:
            mlflow.log_param("pipeline_status", "failed")
            raise e

if __name__ == "__main__":
    main()
  
