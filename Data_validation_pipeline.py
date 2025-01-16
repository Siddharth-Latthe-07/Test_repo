from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig

def run_data_validation(config, train_data):
    data_validation_config = DataValidationConfig(**config["data_validation"])
    data_validation = DataValidation(data_validation_config, train_data)
    data_validation.initiate_data_validation()
  

# src/pipeline/data_validation_pipeline.py
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig

class DataValidationPipeline:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.data_validation = DataValidation(self.config)

    def run(self):
        self.data_validation.validate_data()
        
