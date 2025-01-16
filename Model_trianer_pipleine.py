from src.components.model_trainer import ModelTrainer

class ModelTrainerPipeline:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model_trainer = ModelTrainer(X, y)

    def run(self):
        model = self.model_trainer.train_model()
        return model
        
