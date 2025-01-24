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






from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# Example sentences (unlabelled dataset)
sentences = [
    "She is walking to school", "He went to the store", "They will travel tomorrow",
    "I am reading a book", "We played football yesterday", "He is cooking dinner",
    "They will go to the market", "She studied hard last night", "I am going home now"
]

# Step 1: Generate Custom Word Embeddings with Word2Vec
# Tokenize sentences into words
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# Train a Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, min_count=1, window=5, sg=1)

# Generate sentence embeddings by averaging word embeddings
def sentence_to_embedding(sentence):
    words = sentence.lower().split()
    embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(word2vec_model.vector_size)

sentence_embeddings = np.array([sentence_to_embedding(sentence) for sentence in sentences])

# Step 2: Dimensionality Reduction
svd = TruncatedSVD(n_components=10, random_state=42)
reduced_embeddings = svd.fit_transform(sentence_embeddings)

# Normalize embeddings
normalized_embeddings = normalize(reduced_embeddings)

# Step 3: Clustering with Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
clusters = gmm.fit_predict(normalized_embeddings)

# Evaluate clustering
sil_score = silhouette_score(normalized_embeddings, clusters)
db_score = davies_bouldin_score(normalized_embeddings, clusters)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

# Step 4: Analyze Clusters
for i in range(3):  # Assuming 3 clusters
    print(f"\nCluster {i}:")
    for j, sentence in enumerate(sentences):
        if clusters[j] == i:
            print(f"- {sentence}")






from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np

# Example sentences (unlabelled dataset)
sentences = [
    "She is walking to school", "He went to the store", "They will travel tomorrow",
    "I am reading a book", "We played football yesterday", "He is cooking dinner",
    "They will go to the market", "She studied hard last night", "I am going home now"
]

# Step 1: Feature Extraction with TF-IDF
vectorizer = TfidfVectorizer()
tfidf_features = vectorizer.fit_transform(sentences)

# Step 2: Dimensionality Reduction
svd = TruncatedSVD(n_components=10, random_state=42)
reduced_features = svd.fit_transform(tfidf_features)

# Normalize embeddings
normalized_features = normalize(reduced_features)

# Step 3: Clustering with Agglomerative Clustering
clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
clusters = clustering.fit_predict(normalized_features)

# Evaluate clustering
sil_score = silhouette_score(normalized_features, clusters)
db_score = davies_bouldin_score(normalized_features, clusters)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

# Step 4: Analyze Clusters
for i in range(3):  # Assuming 3 clusters
    print(f"\nCluster {i}:")
    for j, sentence in enumerate(sentences):
        if clusters[j] == i:
            print(f"- {sentence}")







from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np

# Example sentences (unlabelled dataset)
sentences = [
    "She is walking to school", "He went to the store", "They will travel tomorrow",
    "I am reading a book", "We played football yesterday", "He is cooking dinner",
    "They will go to the market", "She studied hard last night", "I am going home now"
]

# Step 1: Feature Extraction with TF-IDF (with bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use unigrams and bigrams
tfidf_features = vectorizer.fit_transform(sentences)

# Step 2: Dimensionality Reduction
svd = TruncatedSVD(n_components=10, random_state=42)
reduced_features = svd.fit_transform(tfidf_features)

# Normalize features
normalized_features = normalize(reduced_features)

# Step 3: Clustering with Spectral Clustering
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
clusters = spectral.fit_predict(normalized_features)

# Evaluate clustering
sil_score = silhouette_score(normalized_features, clusters)
db_score = davies_bouldin_score(normalized_features, clusters)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

# Step 4: Analyze Clusters
for i in range(3):  # Assuming 3 clusters
    print(f"\nCluster {i}:")
    for j, sentence in enumerate(sentences):
        if clusters[j] == i:
            print(f"- {sentence}")
            







from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Example sentences (unlabelled dataset)
sentences = [
    "She is walking to school", "He went to the store", "They will travel tomorrow",
    "I am reading a book", "We played football yesterday", "He is cooking dinner",
    "They will go to the market", "She studied hard last night", "I am going home now"
]

# Step 1: Feature Extraction with TF-IDF (with bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use unigrams and bigrams
tfidf_features = vectorizer.fit_transform(sentences)

# Step 2: Dimensionality Reduction
svd = TruncatedSVD(n_components=10, random_state=42)
reduced_features = svd.fit_transform(tfidf_features)

# Normalize features
normalized_features = normalize(reduced_features)

# Step 3: Clustering with KMeans using Cosine Similarity
# Convert features into a similarity matrix
similarity_matrix = cosine_similarity(normalized_features)

# Apply KMeans on the similarity matrix
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(similarity_matrix)

# Evaluate clustering
sil_score = silhouette_score(similarity_matrix, clusters)
db_score = davies_bouldin_score(similarity_matrix, clusters)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

# Step 4: Analyze Clusters
for i in range(3):  # Assuming 3 clusters
    print(f"\nCluster {i}:")
    for j, sentence in enumerate(sentences):
        if clusters[j] == i:
            print(f"- {sentence}")
            
