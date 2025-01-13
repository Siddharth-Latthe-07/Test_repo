import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load the dataset
file_path = "path/to/your/dataset.xlsx"
df = pd.read_excel(file_path)

# Clean the data
df["CLEANED_SENTENCE"] = df["CLEANED_SENTENCE"].astype(str)

def clean_text(text):
    """Cleans the given text by stripping whitespace and converting to lowercase."""
    return text.strip().lower()

df["CLEANED_SENTENCE"] = df["CLEANED_SENTENCE"].apply(clean_text)

# Map tenses to numerical labels (assumed in dataset)
# Ensure your dataset has correctly labeled examples for these categories
label_mapping = {"past": 0, "present": 1, "present_continuous": 2, "future": 3}
df["label"] = df["Detected_Tense"].map(label_mapping)
df.dropna(subset=["label"], inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["CLEANED_SENTENCE"], df["label"], test_size=0.2, random_state=42
)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(label_mapping))
y_test = to_categorical(y_test, num_classes=len(label_mapping))

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting features for better performance
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Build the deep learning model
model = Sequential([
    Dense(128, input_dim=X_train_tfidf.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_mapping), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_tfidf, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_tfidf, y_test)
print("Test Accuracy:", accuracy)

# Make predictions
y_pred = model.predict(X_test_tfidf)
y_pred_labels = y_pred.argmax(axis=1)
y_test_labels = y_test.argmax(axis=1)

# Evaluate predictions
print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels, target_names=list(label_mapping.keys())))
