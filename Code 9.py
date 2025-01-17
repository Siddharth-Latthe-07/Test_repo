import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Display the dataset structure
print(f"Dataset loaded with {len(df)} rows.")
print(df.head())

# Check if the dataset has a 'text' column
if 'text' not in df.columns:
    raise ValueError("The dataset must have a 'text' column.")

# Ensure no null values in the 'text' column
df = df.dropna(subset=['text']).reset_index(drop=True)

# 2. Preprocessing Function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove numbers and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(preprocess_text)

# 3. Manual Labeling
# Create a label column if it doesn't exist
if 'label' not in df.columns:
    df['label'] = np.nan

# Define possible labels
labels = {
    '0': 'present',
    '1': 'past',
    '2': 'future',
    '3': 'present_continuous'
}

print("Starting manual labeling process...")
print("Labels:")
for key, value in labels.items():
    print(f"{key}: {value}")

for i, row in df.iterrows():
    if pd.isna(row['label']):
        print(f"\nSentence: {row['text']}")
        label = input(f"Enter label (0: present, 1: past, 2: future, 3: present_continuous): ").strip()
        while label not in labels.keys():
            print("Invalid input. Please enter a valid label.")
            label = input(f"Enter label (0: present, 1: past, 2: future, 3: present_continuous): ").strip()
        df.at[i, 'label'] = label

print("\nLabeling completed!")

# Map numeric labels to textual labels
df['label'] = df['label'].map(labels)

# Save the labeled dataset
df.to_excel("labeled_dataset.xlsx", index=False)
print("\nLabeled dataset saved to 'labeled_dataset.xlsx'.")

# 4. Split the Data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# 5. Text Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# 7. Evaluate the Model
y_pred = model.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. Test on Unseen Data
def classify_text(text):
    preprocessed_text = preprocess_text(text)
    text_vec = vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_vec)
    return prediction[0]

# Example usage
unseen_texts = ["I am writing a report.", "She will go to school tomorrow.", "He played football yesterday."]
for text in unseen_texts:
    print(f"Text: {text} -> Predicted Tense: {classify_text(text)}")
    
