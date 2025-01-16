import yaml
from src.entity.config_entity import DataIngestionConfig

class Configuration:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as file:
            self.config_data = yaml.load(file, Loader=yaml.FullLoader)
    
    def get_data_ingestion_config(self):
        data_ingestion_config = self.config_data['data_ingestion']
        return DataIngestionConfig(data_ingestion_config['raw_data_dir'], data_ingestion_config['file_name'])
        
