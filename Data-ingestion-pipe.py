from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

def run_data_ingestion(config):
    data_ingestion_config = DataIngestionConfig(**config["data_ingestion"])
    data_ingestion = DataIngestion(data_ingestion_config)
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    return train_data, test_data





# src/pipeline/data_ingestion_pipeline.py
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

class DataIngestionPipeline:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.data_ingestion = DataIngestion(self.config)

    def run(self):
        self.data_ingestion.ingest_data()
        
