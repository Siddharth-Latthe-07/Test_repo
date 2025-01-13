import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
file_path = "path/to/your/dataset.xlsx"
df = pd.read_excel(file_path)

# Clean the data
df["CLEANED_SENTENCE"] = df["CLEANED_SENTENCE"].astype(str)

def clean_text(text):
    """Cleans the given text by stripping whitespace and converting to lowercase."""
    return text.strip().lower()

df["CLEANED_SENTENCE"] = df["CLEANED_SENTENCE"].apply(clean_text)

# Map tenses to numerical labels
label_mapping = {"past": 0, "present": 1, "present_continuous": 2, "future": 3}
df["label"] = df["CLEANED_SENTENCE"].apply(
    lambda x: label_mapping[x.split("_")[-1]] if any(k in x for k in label_mapping) else np.nan
)
df.dropna(subset=["label"], inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["CLEANED_SENTENCE"], df["label"], test_size=0.2, random_state=42
)

# Tokenize sentences and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = max(len(seq) for seq in X_train_seq)
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding="post")

# Train a Gradient Boosting Classifier
clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
clf.fit(X_train_padded, y_train)

# Make predictions
y_pred = clf.predict(X_test_padded)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())))
