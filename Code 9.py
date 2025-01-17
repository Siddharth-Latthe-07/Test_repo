import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Ensure the dataset contains the required column
if 'CLEANED_SENTENCE' not in df.columns:
    raise ValueError("The dataset must have a 'CLEANED_SENTENCE' column.")

# Check the dataset structure
print("First 5 rows of the dataset:")
print(df.head())

# 2. Manual Labeling
# Add a 'LABEL' column with default value None
df['LABEL'] = None

# Display the first 10 rows for manual labeling
print("\nManual labeling required for the following sentences:")
print(df[['CLEANED_SENTENCE']].head(10))  # Adjust number if needed

# Assign labels to the displayed sentences manually
# Labels: 'present', 'past', 'future', 'present_continuous'
manual_labels = [
    'present',           # Replace with your label for the 1st sentence
    'past',              # Replace with your label for the 2nd sentence
    'future',            # Replace with your label for the 3rd sentence
    'present_continuous',# Replace with your label for the 4th sentence
    'present'            # Replace with your label for the 5th sentence
]

# Update the 'LABEL' column with these labels
df.loc[:len(manual_labels)-1, 'LABEL'] = manual_labels

# Verify that labels have been assigned
print("\nDataset after manual labeling:")
print(df.head(10))

# 3. Continue labeling (if required)
# You can label more rows or implement a loop for iterative labeling.

# Drop rows without labels to use only labeled data for training
df = df.dropna(subset=['LABEL'])

# 4. Map Text Labels to Numeric
label_mapping = {'present': 0, 'past': 1, 'future': 2, 'present_continuous': 3}
df['LABEL'] = df['LABEL'].map(label_mapping)

# 5. Preprocess Text Data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove numbers and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['CLEANED_SENTENCE'] = df['CLEANED_SENTENCE'].apply(preprocess_text)

# 6. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    df['CLEANED_SENTENCE'], 
    df['LABEL'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['LABEL']
)

# 7. Text Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# 9. Evaluate the Model
y_pred = model.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 10. Test on Unseen Data
def classify_text(text):
    preprocessed_text = preprocess_text(text)
    text_vec = vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_vec)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    return reverse_mapping[prediction[0]]

# Example usage with unseen sentences
unseen_sentences = [
    "I am playing football.", 
    "She went to the market.", 
    "We will visit the zoo tomorrow."
]
for sentence in unseen_sentences:
    print(f"Sentence: {sentence} -> Predicted Tense: {classify_text(sentence)}")
    
