from sklearn.metrics import accuracy_score, classification_report

class ModelEval:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self):
        """Evaluate the model and return performance metrics."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
