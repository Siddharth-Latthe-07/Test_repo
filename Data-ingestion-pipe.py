from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

class DataIngestionPipeline:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.data_ingestion = DataIngestion(config)

    def run(self):
        data = self.data_ingestion.ingest_data()
        return data
        
