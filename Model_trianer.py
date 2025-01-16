import joblib
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self, config):
        self.model_output_path = config["model_trainer"]["model_output_path"]
        self.model_type = config["model_trainer"]["model_type"]
        self.n_estimators = config["model_trainer"]["n_estimators"]
        self.random_state = config["model_trainer"]["random_state"]

    def train_model(self, X_train, y_train):
        model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        model.fit(X_train, y_train)
        joblib.dump(model, self.model_output_path)
        return model

    def run(self, X_train, y_train):
        model = self.train_model(X_train, y_train)
        return model
      
