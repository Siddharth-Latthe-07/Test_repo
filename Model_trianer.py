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
        
