from transformers import pipeline
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re

# Load Dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
sheet_name = "Sheet1"  # Replace with the sheet name if needed
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Data Cleaning Function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

data['CLEANED_SENTENCE'] = data['CLEANED_SENTENCE'].apply(clean_text)

# Tokenization and TF-IDF Vectorization
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
X = vectorizer.fit_transform(data['CLEANED_SENTENCE'])

# Dimensionality Reduction using PCA
pca = PCA(n_components=0.95, random_state=42)  # Retain 95% variance
X_reduced = pca.fit_transform(X.toarray())

# Clustering to Assign Initial Labels
n_clusters = 4  # Number of tenses: Present, Past, Future, Present Continuous
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_reduced)
tense_labels = ["Present", "Past", "Future", "Present Continuous"]

# Add labels to the dataset
data['LABEL'] = labels

# Balancing Dataset
smote = SMOTE(sampling_strategy='not majority', random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_reduced, data['LABEL'])
X_balanced, y_balanced = under_sampler.fit_resample(X_balanced, y_balanced)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Model Selection with Regularization
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,  # Regularization to prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_
print(f"\nBest Model: {best_model}")

# Cross-Validation
cv_scores = cross_val_score(best_model, X_balanced, y_balanced, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Model Evaluation
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=tense_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=tense_labels, 
            yticklabels=tense_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the Best Model and Vectorizer
joblib.dump(best_model, "best_tense_classifier_rf_model.pkl")
joblib.dump(vectorizer, "best_vectorizer_rf_model.pkl")
joblib.dump(pca, "pca_transformer.pkl")
print("\nBest Model, PCA Transformer, and Vectorizer Saved.")

# ------------ USER INPUT PREDICTION ------------

def predict_tense(user_input):
    doc = nlp(user_input)
    preprocessed_input = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    X_new = vectorizer.transform([preprocessed_input])
    X_new_reduced = pca.transform(X_new.toarray())
    prediction = best_model.predict(X_new_reduced)
    predicted_tense = tense_labels[prediction[0]]
    print(f"User Input: {user_input}")
    print(f"Predicted Tense: {predicted_tense}\n")

# Get user input and predict tense
while True:
    user_input = input("Enter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    predict_tense(user_input)
