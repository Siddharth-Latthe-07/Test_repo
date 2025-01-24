from transformers import pipeline
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

# Initialize Hugging Face pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["Present", "Past", "Future", "Present Continuous"]

# Function for zero-shot tense classification
def classify_tense(sentence):
    result = classifier(sentence, candidate_labels)
    return result['labels'][0]  # Return top predicted tense

# Load Dataset
file_path = "your_dataset.xlsx"
sheet_name = "Sheet1"
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Clean and preprocess sentences
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

data['CLEANED_SENTENCE'] = data['CLEANED_SENTENCE'].apply(clean_text)

# Tokenization and TF-IDF Vectorization
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
X = vectorizer.fit_transform(data['CLEANED_SENTENCE'])

# Clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)
data['LABEL'] = labels

# Classify Tense using Hugging Face model
data['CLASSIFIED_TENSE'] = data['CLEANED_SENTENCE'].apply(classify_tense)

# Compare clustering results with classified tenses
comparison = data[['CLEANED_SENTENCE', 'LABEL', 'CLASSIFIED_TENSE']]
print(comparison.head())

# Analyze discrepancies
mismatches = data[data['LABEL'] != data['CLASSIFIED_TENSE']]
print("\nMismatched Results:")
print(mismatches)
