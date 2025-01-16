from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig

def run_data_transformation(config, train_data, test_data):
    data_transformation_config = DataTransformationConfig(**config["data_transformation"])
    data_transformation = DataTransformation(data_transformation_config)
    X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(train_data, test_data)
    return X_train, X_test, y_train, y_test



# src/pipeline/data_transformation_pipeline.py
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig

class DataTransformationPipeline:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.data_transformation = DataTransformation(self.config)

    def run(self):
        self.data_transformation.transform_data()
        
