import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import joblib
from transformers import pipeline

# Start MLflow experiment
mlflow.set_experiment("tense_classification_experiment")

# Load Dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
sheet_name = "Sheet1"  # Replace with the sheet name if needed
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Assuming your dataset has a column 'CLEANED_SENTENCE'
sentences = data['CLEANED_SENTENCE']

# Tokenization and TF-IDF Vectorization
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
X = vectorizer.fit_transform(sentences)

# Clustering to Assign Initial Labels
n_clusters = 4  # Number of tenses: Present, Past, Future, Present Continuous
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)
tense_labels = ["Present", "Past", "Future", "Present Continuous"]

# Add labels to the dataset
data['LABEL'] = labels

# Augmentation for Underrepresented Classes
class_counts = data['LABEL'].value_counts()
print("Class distribution before augmentation:")
print(class_counts)

# Initialize a paraphrasing pipeline
paraphraser = pipeline("text2text-generation", model="t5-small", device=0)  # Adjust model/device as needed

augmented_data = pd.DataFrame({'sentence': sentences, 'label': labels})

while True:
    class_counts = augmented_data['label'].value_counts()
    min_count = class_counts.min()
    majority_count = class_counts.max()

    if min_count == majority_count:
        break

    for label, count in class_counts.items():
        if count < majority_count:
            sentences_to_augment = augmented_data[augmented_data['label'] == label]['sentence']
            augment_count = majority_count - count

            if augment_count > 0:
                augmented_sentences = []
                for sentence in sentences_to_augment.sample(n=min(len(sentences_to_augment), augment_count), random_state=42):
                    # Generate paraphrased variations
                    try:
                        paraphrased = paraphraser(sentence, max_length=50, num_return_sequences=1)[0]['generated_text']
                        augmented_sentences.append(paraphrased)
                    except Exception as e:
                        print(f"Error in paraphrasing: {e}")
                        continue

                # Create new DataFrame for augmented sentences
                augment_df = pd.DataFrame({'sentence': augmented_sentences, 'label': label})
                augmented_data = pd.concat([augmented_data, augment_df], ignore_index=True)

# Shuffle and Verify Class Distribution
augmented_data = augmented_data.sample(frac=1, random_state=42).reset_index(drop=True)
new_class_counts = augmented_data['label'].value_counts()
print("Class distribution after augmentation:")
print(new_class_counts)

# Tokenization and TF-IDF Vectorization
X = vectorizer.fit_transform(augmented_data['sentence'])
y = augmented_data['label']

# Train-Test Split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Model Selection (Random Forest)
model = RandomForestClassifier(random_state=42)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Start MLflow run for the model training
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("train_test_split_ratio", test_size)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Log best parameters from GridSearchCV
    mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
    
    # Best Model and Cross-validation
    best_model = grid_search.best_estimator_
    cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    mlflow.log_metric("mean_cv_accuracy", cross_val_scores.mean())

    # Model Evaluation
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=tense_labels))

    # Log classification report and confusion matrix
    mlflow.log_metric("accuracy", np.mean(cross_val_scores))  # Log accuracy

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=tense_labels, yticklabels=tense_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Log confusion matrix plot as artifact
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Save the best model as an artifact
    joblib.dump(best_model, "best_tense_classifier_rf_model.pkl")
    joblib.dump(vectorizer, "best_vectorizer_rf_model.pkl")
    mlflow.log_artifact("best_tense_classifier_rf_model.pkl")
    mlflow.log_artifact("best_vectorizer_rf_model.pkl")

    print("\nBest Model and Vectorizer Saved.")

# ------------ USER INPUT PREDICTION ------------

def predict_tense(user_input):
    # Preprocess the user input
    doc = nlp(user_input)
    preprocessed_input = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    
    # Vectorize the preprocessed sentence
    X_new = vectorizer.transform([preprocessed_input])
    
    # Predict the tense
    prediction = best_model.predict(X_new)
    predicted_tense = tense_labels[prediction[0]]
    
    # Display the result
    print(f"User Input: {user_input}")
    print(f"Predicted Tense: {predicted_tense}\n")

# Get user input and predict tense
while True:
    user_input = input("Enter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    predict_tense(user_input)
