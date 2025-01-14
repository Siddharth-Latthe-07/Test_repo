import pandas as pd
import numpy as np
import spacy
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical

# Step 1: Load Dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
sheet_name = "Sheet1"  # Replace with the sheet name if needed
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Assuming your dataset has a column 'CLEANED_SENTENCE'
sentences = data['CLEANED_SENTENCE']

# Step 2: Embedding Extraction using spaCy
nlp = spacy.load("en_core_web_sm")

def get_sentence_embedding(sentence):
    doc = nlp(sentence)
    return np.mean([token.vector for token in doc if token.has_vector], axis=0)

sentence_embeddings = np.array([get_sentence_embedding(sentence) for sentence in sentences])

# Step 3: Clustering (Pseudo-Labeling)
n_clusters = 4  # Number of tenses
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(sentence_embeddings)
data['Cluster'] = clusters

# Map clusters to tenses based on manual inspection (adjust as necessary)
cluster_to_tense = {
    0: 'Present', 
    1: 'Past', 
    2: 'Future', 
    3: 'Present Continuous'
}
data['Tense'] = data['Cluster'].map(cluster_to_tense)

# Step 4: Prepare Data for Deep Learning
# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, clusters, test_size=0.2, random_state=42)
y_train_categorical = to_categorical(y_train, num_classes=n_clusters)
y_test_categorical = to_categorical(y_test, num_classes=n_clusters)

# Step 5: Build Deep Learning Model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dense(n_clusters, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 6: Train the Model
history = model.fit(X_train, y_train_categorical, validation_split=0.2, epochs=10, batch_size=32, verbose=1)

# Step 7: Evaluate the Model
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[cluster_to_tense[i] for i in range(n_clusters)]))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[cluster_to_tense[i] for i in range(n_clusters)], 
            yticklabels=[cluster_to_tense[i] for i in range(n_clusters)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 8: Save the Model and Tokenizer
model.save("tense_classifier_model.h5")
with open("tokenizer.json", "w") as f:
    f.write(tokenizer.to_json())
print("\nModel and tokenizer saved.")

# Step 9: Interactive Input for Dynamic Classification
print("\nEnter a sentence to classify:")
user_input = input()
user_sequence = tokenizer.texts_to_sequences([user_input])
user_padded_sequence = pad_sequences(user_sequence, maxlen=max_sequence_length, padding='post')
user_pred_prob = model.predict(user_padded_sequence)
user_cluster = np.argmax(user_pred_prob, axis=1)[0]
user_tense = cluster_to_tense.get(user_cluster, "Unknown")

print(f"\nPredicted Tense for '{user_input}': {user_tense}")

# Step 10: Save Results with Tense Information
output_path = "classified_dataset_with_tenses.xlsx"
data.to_excel(output_path, index=False)
print(f"\nClassified dataset with tenses saved to {output_path}")
