from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_model(self):
        """Train the Random Forest model."""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Save the trained model
        joblib.dump(model, 'random_forest_model.pkl')
        print("Model training completed.")
        
        return model









from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import mlflow

class ModelTrainer:
    def __init__(self, X, y, model_training_config, output_config):
        self.X = X
        self.y = y
        self.config = model_training_config
        self.output_config = output_config

    def train_model(self):
        """Train the model and log with MLflow."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.config['test_size'], random_state=self.config['random_state']
        )
        
        # Initialize the model
        model = RandomForestClassifier(
            random_state=self.config['random_state'],
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth']
        )
        
        # Train the model
        model.fit(X_train, y_train)
        print("Model training completed.")
        
        # Save the trained model
        model_save_path = f"{self.output_config['model_save_path']}/{self.output_config['model_file_name']}"
        joblib.dump(model, model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Log model and parameters in MLflow
        mlflow.log_param("n_estimators", self.config['n_estimators'])
        mlflow.log_param("max_depth", self.config['max_depth'])
        mlflow.log_param("random_state", self.config['random_state'])
        mlflow.sklearn.log_model(model, "random_forest_model")

        return model
        
