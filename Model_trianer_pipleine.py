from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig

def run_model_trainer(config, X_train, y_train):
    model_trainer_config = ModelTrainerConfig(**config["model_trainer"])
    model_trainer = ModelTrainer(model_trainer_config)
    model = model_trainer.initiate_model_training(X_train, y_train)
    return model
  
