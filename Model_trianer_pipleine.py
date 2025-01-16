from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig

def run_model_trainer(config, X_train, y_train):
    model_trainer_config = ModelTrainerConfig(**config["model_trainer"])
    model_trainer = ModelTrainer(model_trainer_config)
    model = model_trainer.initiate_model_training(X_train, y_train)
    return model





# src/pipeline/model_trainer_pipeline.py
from src/components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainerPipeline:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model_trainer = ModelTrainer(self.config)

    def run(self):
        self.model_trainer.train_model()
        
