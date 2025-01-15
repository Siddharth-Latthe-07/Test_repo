import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import joblib

# Load Dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
sheet_name = "Sheet1"  # Replace with your sheet name if necessary
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocessing with spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct or token.text in ['is', 'will', 'was', 'am']
    ]
    return ' '.join(tokens)

data['cleaned_sentence'] = data['CLEANED_SENTENCE'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(data['cleaned_sentence'])

# Label Preparation (Supervised Learning)
n_clusters = 4  # Number of tenses: Present, Past, Future, Present Continuous
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['LABEL'] = kmeans.fit_predict(X)
tense_labels = ["Present", "Past", "Future", "Present Continuous"]
data['LABEL'] = data['LABEL'].map(dict(enumerate(tense_labels)))

# Oversampling Using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, data['LABEL'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Random Forest Model with Hyperparameter Tuning
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Cross-Validation
skf = StratifiedKFold(n_splits=5)
cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring='accuracy')
print("\nCross-validation scores:", cross_val_scores)
print("Mean CV accuracy:", cross_val_scores.mean())

# Model Evaluation
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=tense_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=tense_labels, yticklabels=tense_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save Best Model and Vectorizer
joblib.dump(best_model, "best_tense_classifier_rf_model.pkl")
joblib.dump(vectorizer, "best_vectorizer_rf_model.pkl")
print("\nBest Model and Vectorizer Saved.")

# Clustering Verification (Unsupervised Learning)
predicted_clusters = kmeans.predict(X_test)
print("\nClustering Verification Results:")
print(confusion_matrix(y_test.map(dict(zip(tense_labels, range(n_clusters)))), predicted_clusters))
