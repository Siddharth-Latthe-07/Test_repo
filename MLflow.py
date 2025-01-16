import mlflow
import mlflow.sklearn
from src.config.configuration import Configuration
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_validation_pipeline import DataValidationPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.pipeline.model_eval_pipeline import ModelEvalPipeline

if __name__ == "__main__":
    # Start MLflow experiment
    mlflow.set_tracking_uri("file:///mlruns")  # Local directory for MLflow logs
    mlflow.set_experiment("ML Pipeline Experiment")

    with mlflow.start_run() as run:
        # Load the config
        config = Configuration(config_file="config/config.yaml")

        # Log general pipeline parameters
        mlflow.log_param("config_file", "config/config.yaml")

        # Data Ingestion
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion_pipeline = DataIngestionPipeline(config=data_ingestion_config)
        data = data_ingestion_pipeline.run()
        mlflow.log_param("raw_data_path", data_ingestion_config.raw_data_dir)
        mlflow.log_metric("data_ingested_rows", len(data))

        # Data Validation
        data_validation_pipeline = DataValidationPipeline(data=data)
        validated_data = data_validation_pipeline.run()
        mlflow.log_metric("validated_data_rows", len(validated_data))

        # Data Transformation
        data_transformation_pipeline = DataTransformationPipeline(data=validated_data)
        X, y, vectorizer = data_transformation_pipeline.run()
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_metric("transformed_features", X.shape[1])

        # Save vectorizer as an artifact
        vectorizer_file = "artifacts/vectorizer.pkl"
        vectorizer.save(vectorizer_file)
        mlflow.log_artifact(vectorizer_file)

        # Model Training
        model_trainer_pipeline = ModelTrainerPipeline(X, y)
        model = model_trainer_pipeline.run()
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", config.model_trainer.n_estimators)

        # Save the trained model
        model_file = "artifacts/model.pkl"
        model.save(model_file)
        mlflow.log_artifact(model_file)

        # Model Evaluation
        model_eval_pipeline = ModelEvalPipeline(model, X, y)
        metrics = model_eval_pipeline.run()
        mlflow.log_metrics(metrics)

        print(f"MLflow Run ID: {run.info.run_id}")
        
