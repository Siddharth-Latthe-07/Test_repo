import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn

# Step 1: Load Dataset
file_path = "augmented_data.xlsx"
data = pd.read_excel(file_path)

# Data Columns
sentences = data['sentence']
labels = data['label']

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)
y = labels

# Step 3: Train-Test Split
train_test_ratio = 0.8
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_test_ratio), random_state=random_state)

# Step 4: Train a Model
model = RandomForestClassifier(random_state=random_state, n_estimators=100)

# Configure MLflow
mlflow.set_tracking_uri("file:///mlruns")  # Local directory for tracking
experiment_name = "Tense Classification Experiment"
mlflow.set_experiment(experiment_name)

# Start MLflow Run
with mlflow.start_run():
    # Log Parameters
    mlflow.log_param("train_test_ratio", train_test_ratio)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save model
    model_file = "tense_classifier_model.pkl"
    vectorizer_file = "tfidf_vectorizer.pkl"
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    
    # Log Artifacts
    mlflow.log_artifact(model_file)
    mlflow.log_artifact(vectorizer_file)
    
    # Evaluate the Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Classification Report
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Log the report as an artifact
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact("classification_report.txt")
    
    print("\nExperiment successfully logged with MLflow.")

# After the script, you can check the results using the MLflow UI
# Run the following command in your terminal: `mlflow ui`
