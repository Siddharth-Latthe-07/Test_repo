from sklearn.metrics import accuracy_score, classification_report

class ModelEvaluation:
    def __init__(self, config):
        self.report_file_path = config["model_evaluation"]["report_file_path"]

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        with open(self.report_file_path, "w") as f:
            f.write(report)
        print("Evaluation Report:\n", report)
        return accuracy_score(y_test, y_pred)

    def run(self, model, X_test, y_test):
        accuracy = self.evaluate(model, X_test, y_test)
        return accuracy
      
