import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re

# 1. Load the dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Check the dataset structure
print(df.head())

# 2. Manual Labeling (if required)
# Add a 'label' column to the dataset manually. Labels: 'present', 'past', 'future', 'present_continuous'
# Uncomment below if needed
# df['label'] = ['present', 'past', 'future', 'present_continuous', ...]  # Add labels manually

if 'label' not in df.columns:
    raise ValueError("Dataset must have a 'label' column for supervised learning.")

# Ensure no null values in the dataset
df.dropna(subset=['text', 'label'], inplace=True)

# 3. Preprocessing Function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove numbers and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(preprocess_text)

# 4. Encode Labels
label_mapping = {'present': 0, 'past': 1, 'future': 2, 'present_continuous': 3}
df['label'] = df['label'].map(label_mapping)

# 5. Split the Data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# 6. Text Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# 8. Evaluate the Model
y_pred = model.predict(X_test_vec)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 9. Test on Unseen Data
def classify_text(text):
    preprocessed_text = preprocess_text(text)
    text_vec = vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_vec)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    return reverse_mapping[prediction[0]]

# Example usage
unseen_texts = ["I am writing a report.", "She will go to school tomorrow.", "He played football yesterday."]
for text in unseen_texts:
    print(f"Text: {text} -> Predicted Tense: {classify_text(text)}")
