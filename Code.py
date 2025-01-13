import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load the dataset
file_path = "path/to/your/dataset.xlsx"
df = pd.read_excel(file_path)

# Step 2: Clean the data
df["CLEANED_SENTENCE"] = df["CLEANED_SENTENCE"].astype(str)

def clean_text(text):
    """Cleans the given text by stripping whitespace and converting to lowercase."""
    return text.strip().lower()

df["CLEANED_SENTENCE"] = df["CLEANED_SENTENCE"].apply(clean_text)

# Step 3: Create a small labeled dataset (manual labeling)
labeled_data = {
    "CLEANED_SENTENCE": [
        "I am eating food.",  # present_continuous
        "She went to school.",  # past
        "I will go there tomorrow.",  # future
        "He plays cricket.",  # present
    ],
    "label": [2, 0, 3, 1]
}
labeled_df = pd.DataFrame(labeled_data)

# Combine the labeled dataset with the main dataset
df["label"] = None  # Add a placeholder column for labels

# Step 4: Train a model using the labeled dataset
X_train, X_test, y_train, y_test = train_test_split(
    labeled_df["CLEANED_SENTENCE"], labeled_df["label"], test_size=0.2, random_state=42
)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_tfidf, y_train)

# Step 5: Predict labels for the entire dataset dynamically
df["label"] = clf.predict(vectorizer.transform(df["CLEANED_SENTENCE"]))

# Step 6: Evaluate on test set (optional)
y_pred = clf.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["past", "present", "present_continuous", "future"]))

# Save the labeled dataset to a new file
df.to_excel("labeled_dataset.xlsx", index=False)
print("Labeled dataset saved as 'labeled_dataset.xlsx'")
