from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

def run_data_ingestion(config):
    data_ingestion_config = DataIngestionConfig(**config["data_ingestion"])
    data_ingestion = DataIngestion(data_ingestion_config)
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    return train_data, test_data
  
