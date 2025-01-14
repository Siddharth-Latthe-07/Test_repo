from transformers import pipeline
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
labels = data['LABEL']  # Replace 'LABEL' with your column name for tense labels

# Step 2: Class Distribution Analysis
class_counts = labels.value_counts()
print("Class distribution before augmentation:")
print(class_counts)

# Step 3: Augmentation for Underrepresented Classes
tense_labels = ["Present", "Past", "Future", "Present Continuous"]
max_count = class_counts.max()

# Initialize a paraphrasing pipeline
paraphraser = pipeline("text2text-generation", model="t5-small", device=0)  # Adjust model/device as needed

augmented_data = pd.DataFrame({'sentence': sentences, 'label': labels})

for label, count in class_counts.items():
    sentences_to_augment = augmented_data[augmented_data['label'] == label]['sentence']
    augment_count = max_count - count

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

# Step 4: Shuffle and Verify Class Distribution
augmented_data = augmented_data.sample(frac=1, random_state=42).reset_index(drop=True)
new_class_counts = augmented_data['label'].value_counts()
print("Class distribution after augmentation:")
print(new_class_counts)

# Step 5: Tokenization and TF-IDF Vectorization
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
X = vectorizer.fit_transform(augmented_data['sentence'])
y = augmented_data['label']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Selection (Random Forest)
model = RandomForestClassifier(random_state=42)

# Step 8: Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Step 9: Best Model and Cross-validation
best_model = grid_search.best_estimator_
cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cross_val_scores}")
print(f"Mean CV accuracy: {cross_val_scores.mean()}")

# Step 10: Model Evaluation
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

# Step 11: Save the Best Model and Vectorizer
joblib.dump(best_model, "best_tense_classifier_rf_model.pkl")
joblib.dump(vectorizer, "best_vectorizer_rf_model.pkl")
print("\nBest Model and Vectorizer Saved.")
