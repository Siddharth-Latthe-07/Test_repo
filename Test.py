import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example sentences (label them manually)
sentences = ["She is walking to school", "He went to the store", "They will travel tomorrow"]
labels = [1, 0, 2]  # 0: Past, 1: Present, 2: Future

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(sentences)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Past, Present, Future
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
labels = np.array(labels)
model.fit(padded_sequences, labels, epochs=10, batch_size=2)

# Prediction
new_sentence = ["I will call you later"]
new_sequence = tokenizer.texts_to_sequences(new_sentence)
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_length, padding='post')
prediction = model.predict(new_padded_sequence)
print("Predicted class:", np.argmax(prediction))






import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example sentences and labels
sentences = ["She is walking to school", "He went to the store", "They will travel tomorrow"]
labels = [1, 0, 2]  # 0: Past, 1: Present, 2: Future

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(sentences)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# CNN Model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length),
    Conv1D(filters=64, kernel_size=3, activation='relu'),  # Extract n-gram features
    GlobalMaxPooling1D(),  # Reduce sequence to a single vector
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Past, Present, Future
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
labels = np.array(labels)
model.fit(padded_sequences, labels, epochs=10, batch_size=2)

# Prediction
new_sentence = ["I will call you later"]
new_sequence = tokenizer.texts_to_sequences(new_sentence)
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_length, padding='post')
prediction = model.predict(new_padded_sequence)
print("Predicted class:", np.argmax(prediction))  # Output: 2 (Future)







from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# Example sentences (unlabelled dataset)
sentences = [
    "She is walking to school", "He went to the store", "They will travel tomorrow",
    "I am reading a book", "We played football yesterday", "He is cooking dinner",
    "They will go to the market", "She studied hard last night", "I am going home now"
]

# Step 1: Generate Sentence Embeddings
# Using a pre-trained model from Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentences)

# Step 2: Apply K-Means Clustering
num_clusters = 3  # Assuming 3 clusters for Past, Present, and Future
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(sentence_embeddings)
clusters = kmeans.labels_

# Step 3: Visualize Clusters (Optional)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(sentence_embeddings)

plt.figure(figsize=(8, 6))
for i, label in enumerate(clusters):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label=f"Cluster {label}")
    plt.text(reduced_embeddings[i, 0] + 0.02, reduced_embeddings[i, 1] + 0.02, sentences[i], fontsize=9)
plt.title("Sentence Clusters")
plt.show()

# Step 4: Analyze Clusters
for i in range(num_clusters):
    print(f"\nCluster {i}:")
    for j, sentence in enumerate(sentences):
        if clusters[j] == i:
            print(f"- {sentence}")
            
