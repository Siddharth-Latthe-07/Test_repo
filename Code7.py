from transformers import pipeline
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re

# Load Dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
sheet_name = "Sheet1"  # Replace with the sheet name if needed
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Assuming your dataset has a column 'CLEANED_SENTENCE'
sentences = data['CLEANED_SENTENCE']

# Data Cleaning Function
def clean_text(text):
    """
    Cleans the given text by removing unwanted characters, extra spaces,
    and converting to lowercase.
    """
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
X = vectorizer.fit_transform(sentences)

# Clustering to Assign Initial Labels
n_clusters = 4  # Number of tenses: Present, Past, Future, Present Continuous
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)
tense_labels = ["Present", "Past", "Future", "Present Continuous"]

# Add labels to the dataset
data['LABEL'] = labels

# Balancing Dataset
class_counts = data['LABEL'].value_counts()
print("\nClass distribution before balancing:")
print(class_counts)

# Over-sampling and Under-sampling
smote = SMOTE(sampling_strategy='not majority', random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)

X_balanced, y_balanced = smote.fit_resample(X, data['LABEL'])
X_balanced, y_balanced = under_sampler.fit_resample(X_balanced, y_balanced)

balanced_data = pd.DataFrame(X_balanced.toarray(), columns=vectorizer.get_feature_names_out())
balanced_data['LABEL'] = y_balanced

print("\nClass distribution after balancing:")
print(balanced_data['LABEL'].value_counts())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Model Selection (Random Forest)
model = RandomForestClassifier(random_state=42)

# Hyperparameter Tuning using GridSearchCV with early stopping functionality
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best Model and Cross-validation
best_model = grid_search.best_estimator_
print(f"\nBest Model: {best_model}")

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
    
