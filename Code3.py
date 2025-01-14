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

# Step 1: Load Dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
sheet_name = "Sheet1"  # Replace with the sheet name if needed
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Assuming your dataset has a column 'CLEANED_SENTENCE'
sentences = data['CLEANED_SENTENCE']

# Step 2: Tokenization and TF-IDF Vectorization
nlp = spacy.load("en_core_web_sm")

# Tokenization using spaCy
def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

# Vectorizing sentences using TF-IDF
vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
X = vectorizer.fit_transform(sentences)

# Step 3: Clustering (Optional, for initial label generation)
n_clusters = 4  # Number of tenses: Present, Past, Future, Present Continuous
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y = kmeans.fit_predict(X)

# Augment Dataset to Balance Classes
# Count the support for each class
class_counts = pd.Series(y).value_counts()
max_count = class_counts.max()

# Mapping of cluster IDs to labels
tense_labels = ["Present", "Past", "Future", "Present Continuous"]

# Create a new DataFrame for augmentation
augmented_data = pd.DataFrame({'sentence': sentences, 'label': y})

# Augment the dataset
for cluster, count in class_counts.items():
    # Calculate the number of sentences to add
    sentences_to_add = max_count - count
    
    # Filter sentences for the current class
    cluster_sentences = augmented_data[augmented_data['label'] == cluster]['sentence']
    
    # If there are sentences to duplicate
    if sentences_to_add > 0:
        # Duplicate sentences to match the majority class
        duplicates = cluster_sentences.sample(sentences_to_add, replace=True, random_state=42)
        duplicates_df = pd.DataFrame({'sentence': duplicates, 'label': cluster})
        augmented_data = pd.concat([augmented_data, duplicates_df], ignore_index=True)

# Shuffle the augmented dataset
augmented_data = augmented_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Verify the new class distribution
new_class_counts = augmented_data['label'].value_counts()
print("Class distribution after augmentation:")
print(new_class_counts)

# Use the augmented dataset for further processing
sentences = augmented_data['sentence']
y = augmented_data['label']

# Step 4: Train-Test Split
X = vectorizer.fit_transform(sentences)  # Refit vectorizer on augmented data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection (Random Forest)
model = RandomForestClassifier(random_state=42)

# Step 6: Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [None, 10, 20],  # Max depth of trees
    'min_samples_split': [2, 5, 10],  # Min samples to split internal nodes
    'min_samples_leaf': [1, 2, 4],  # Min samples in leaf nodes
    'max_features': ['auto', 'sqrt', 'log2']  # Maximum features for splitting
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Step 7: Best Parameters from GridSearchCV
print(f"Best Parameters: {grid_search.best_params_}")

# Step 8: Train the Best Model
best_model = grid_search.best_estimator_

# Step 9: Cross-validation for model assessment
cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cross_val_scores}")
print(f"Mean CV accuracy: {cross_val_scores.mean()}")

# Step 10: Model Evaluation
y_pred = best_model.predict(X_test)

# Step 11: Evaluation Metrics
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

# Step 12: Save the Best Model and Vectorizer
joblib.dump(best_model, "best_tense_classifier_rf_model.pkl")
joblib.dump(vectorizer, "best_vectorizer_rf_model.pkl")
print("\nBest Model and Vectorizer Saved.")

# Step 13: Interactive Input for Dynamic Classification
print("\nEnter a sentence to classify:")
user_input = input()
user_input_vectorized = vectorizer.transform([user_input])
user_pred = best_model.predict(user_input_vectorized)

print(f"\nPredicted Tense for '{user_input}': {tense_labels[user_pred[0]]}")