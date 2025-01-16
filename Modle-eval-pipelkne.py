from src.components.model_eval import ModelEval
from sklearn.model_selection import train_test_split

class ModelEvalPipeline:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model_eval = ModelEval(self.model, X_test, y_test)
        model_eval.evaluate_model()
        
