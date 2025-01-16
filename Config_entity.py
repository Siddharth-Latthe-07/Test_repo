class ConfigEntity:
    def __init__(self, config):
        self.data_ingestion = config["data_ingestion"]
        self.data_validation = config["data_validation"]
        self.data_transformation = config["data_transformation"]
        self.model_trainer = config["model_trainer"]
        self.model_evaluation = config["model_evaluation"]
        self.mlflow = config["mlflow"]
      
