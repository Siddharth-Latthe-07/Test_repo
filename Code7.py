from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
sheet_name = "Sheet1"  # Replace with the sheet name if needed
data = pd.read_excel(file_path, sheet_name=sheet_name)

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

# TF-IDF Vectorization with Feature Limit
vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words="english")
X = vectorizer.fit_transform(data['CLEANED_SENTENCE'])
y = data['LABEL']

# Balancing Dataset with SMOTE and RandomUnderSampler
smote = SMOTE(sampling_strategy='not majority', random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
X_balanced, y_balanced = under_sampler.fit_resample(X_balanced, y_balanced)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Random Forest Classifier with Regularization
model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Limit the depth of the trees
    min_samples_split=5,  # Minimum samples required to split a node
    min_samples_leaf=3,   # Minimum samples required at a leaf node
    random_state=42,
    max_features="sqrt",  # Use square root of features for splits
)

# Cross-Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")

# Fit the Model
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())  # Get the sorted labels
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, 
            yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
feature_importances = model.feature_importances_
sorted_indices = feature_importances.argsort()[::-1][:10]
top_features = [vectorizer.get_feature_names_out()[i] for i in sorted_indices]

plt.figure(figsize=(10, 6))
plt.barh(top_features, feature_importances[sorted_indices])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Important Features')
plt.gca().invert_yaxis()
plt.show()
