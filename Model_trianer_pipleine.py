from src.components.model_trainer import ModelTrainer

class ModelTrainerPipeline:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model_trainer = ModelTrainer(X, y)

    def run(self):
        model = self.model_trainer.train_model()
        return model






from src.components.model_trainer import ModelTrainer

class ModelTrainerPipeline:
    def __init__(self, X, y, model_training_config, output_config):
        self.X = X
        self.y = y
        self.model_training_config = model_training_config
        self.output_config = output_config
        self.model_trainer = ModelTrainer(X, y, model_training_config, output_config)

    def run(self):
        model = self.model_trainer.train_model()
        return model
        
