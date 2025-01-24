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








import gensim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np

# Load the pre-trained Word2Vec model from Hugging Face
model_path = "path/to/word2vec-google-news-300.model"  # Replace with your local path
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# Example sentences (unlabelled dataset)
sentences = [
    "She is walking to school", "He went to the store", "They will travel tomorrow",
    "I am reading a book", "We played football yesterday", "He is cooking dinner",
    "They will go to the market", "She studied hard last night", "I am going home now"
]

# Function to get average embedding of the sentence
def get_sentence_embedding(sentence, model):
    words = sentence.split()
    word_embeddings = []
    for word in words:
        try:
            # Get the word embedding for each word in the sentence
            word_embeddings.append(model[word])
        except KeyError:
            # Skip words that aren't in the model's vocabulary
            continue
    # Return the average of all word embeddings in the sentence
    if len(word_embeddings) > 0:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(300)  # Return a zero vector if no words are in the model

# Step 1: Convert sentences to embeddings
sentence_embeddings = np.array([get_sentence_embedding(sentence, word2vec_model) for sentence in sentences])

# Step 2: Dimensionality Reduction (SVD)
svd = TruncatedSVD(n_components=10, random_state=42)
reduced_embeddings = svd.fit_transform(sentence_embeddings)

# Normalize features
normalized_embeddings = normalize(reduced_embeddings)

# Step 3: Clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normalized_embeddings)

# Step 4: Evaluate clustering
sil_score = silhouette_score(normalized_embeddings, clusters)
db_score = davies_bouldin_score(normalized_embeddings, clusters)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

# Step 5: Analyze Clusters
for i in range(3):  # Assuming 3 clusters
    print(f"\nCluster {i}:")
    for j, sentence in enumerate(sentences):
        if clusters[j] == i:
            print(f"- {sentence}")







import gensim.downloader as api
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np

# Load pre-trained Word2Vec model
word2vec_model = api.load('word2Vec-google-news-300')

# Example sentences (unlabelled dataset)
sentences = [
    "She is walking to school", "He went to the store", "They will travel tomorrow",
    "I am reading a book", "We played football yesterday", "He is cooking dinner",
    "They will go to the market", "She studied hard last night", "I am going home now"
]

# Function to get average embedding of the sentence
def get_sentence_embedding(sentence, model):
    words = sentence.split()
    word_embeddings = []
    for word in words:
        try:
            # Get the word embedding for each word in the sentence
            word_embeddings.append(model[word])
        except KeyError:
            # Skip words that aren't in the model's vocabulary
            continue
    # Return the average of all word embeddings in the sentence
    if len(word_embeddings) > 0:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(300)  # Return a zero vector if no words are in the model

# Step 1: Convert sentences to embeddings
sentence_embeddings = np.array([get_sentence_embedding(sentence, word2vec_model) for sentence in sentences])

# Step 2: Dimensionality Reduction (SVD)
svd = TruncatedSVD(n_components=10, random_state=42)
reduced_embeddings = svd.fit_transform(sentence_embeddings)

# Normalize features
normalized_embeddings = normalize(reduced_embeddings)

# Step 3: Clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normalized_embeddings)

# Step 4: Evaluate clustering
sil_score = silhouette_score(normalized_embeddings, clusters)
db_score = davies_bouldin_score(normalized_embeddings, clusters)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

# Step 5: Analyze Clusters
for i in range(3):  # Assuming 3 clusters
    print(f"\nCluster {i}:")
    for j, sentence in enumerate(sentences):
        if clusters[j] == i:
            print(f"- {sentence}")
            
            







import gensim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load Word2Vec model
word2vec_model = gensim.models.KeyedVectors.load('path_to_your_model/word2vec-google-news-300.model')

# Function to generate sentence embeddings by averaging word embeddings
def get_sentence_embedding(sentence, model):
    words = sentence.split()  # Simple tokenization, adjust if needed
    word_embeddings = []
    for word in words:
        if word in model:
            word_embeddings.append(model[word])
    if word_embeddings:
        sentence_embedding = np.mean(word_embeddings, axis=0)  # Mean of word vectors
        return sentence_embedding
    else:
        return np.zeros(model.vector_size)  # Return a zero vector if no word is in the model

# Sample sentences (for testing purposes)
sentences = [
    "She walks to the park.",
    "She walked to the park.",
    "She will walk to the park.",
    "They are running in the field.",
    "They were running in the field.",
    "They will be running in the field."
]

# Generate embeddings for each sentence
sentence_embeddings = []
for sentence in sentences:
    sentence_embeddings.append(get_sentence_embedding(sentence, word2vec_model))

# Convert to numpy array for clustering
sentence_embeddings = np.array(sentence_embeddings)

# Standardize the embeddings (optional, but can help with clustering)
scaler = StandardScaler()
sentence_embeddings_scaled = scaler.fit_transform(sentence_embeddings)

# Apply KMeans clustering
n_clusters = 3  # Number of tenses (past, present, future)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(sentence_embeddings_scaled)

# Predict the clusters
clusters = kmeans.predict(sentence_embeddings_scaled)

# Evaluate clustering
sil_score = silhouette_score(sentence_embeddings_scaled, clusters)
davies_bouldin = davies_bouldin_score(sentence_embeddings_scaled, clusters)

print(f"Silhouette Score: {sil_score}")
print(f"Davies-Bouldin Score: {davies_bouldin}")

# Print cluster labels and their respective sentences
for cluster_num in range(n_clusters):
    print(f"\nCluster {cluster_num}:")
    for i, label in enumerate(clusters):
        if label == cluster_num:
            print(f" - {sentences[i]}")
            







import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Step 1: Load the spacy model
nlp = spacy.load("en_core_web_md")

# Step 2: Load your dataset
df = pd.read_excel('your_dataset.xlsx')

# Step 3: Process the sentences to obtain embeddings
sentences = df['CLEANED SENTENCE'].tolist()

# Step 4: Get sentence embeddings by averaging word vectors
def get_sentence_embedding(sentence, nlp_model):
    doc = nlp_model(sentence)
    # Take the average of word vectors (ignoring words without vectors)
    word_vectors = [token.vector for token in doc if token.has_vector]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average of word vectors
    else:
        return np.zeros(nlp_model.vector_size)  # If no valid word vectors, return zero vector

sentence_embeddings = [get_sentence_embedding(sentence, nlp) for sentence in sentences]

# Step 5: Clustering the sentence embeddings using KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(sentence_embeddings)

# Get the predicted clusters for each sentence
df['Cluster'] = kmeans.labels_

# Step 6: Visualizing the clusters (Optional, using PCA for dimensionality reduction)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(sentence_embeddings)

plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("Clustered Sentences based on Tense")
plt.show()

# Step 7: Evaluate the clustering performance (Silhouette score)
silhouette_avg = silhouette_score(sentence_embeddings, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)

# Step 8: Analyze and label clusters based on tense (manual or semi-automatic)
# Example of viewing sentences in each cluster
for cluster_num in range(4):
    print(f"\nCluster {cluster_num} sentences:")
    cluster_sentences = df[df['Cluster'] == cluster_num]['CLEANED SENTENCE']
    for sentence in cluster_sentences:
        print(sentence)

# Step 9: Save the output to Excel
output_file = 'clustered_sentences_spacy.xlsx'
df.to_excel(output_file, index=False)

print(f"Clustered sentences saved to {output_file}")







import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Step 1: Load the spacy model
nlp = spacy.load("en_core_web_md")

# Step 2: Load your dataset
df = pd.read_excel('your_dataset.xlsx')  # Replace with your actual file path
sentences = df['CLEANED SENTENCE'].tolist()

# Step 3: Function to get sentence embeddings by averaging word vectors
def get_sentence_embedding(sentence, nlp_model):
    doc = nlp_model(sentence)
    word_vectors = [token.vector for token in doc if token.has_vector]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average of word vectors
    else:
        return np.zeros(nlp_model.vector_size)  # If no valid word vectors, return zero vector

# Generate embeddings for all sentences
sentence_embeddings = [get_sentence_embedding(sentence, nlp) for sentence in sentences]

# Step 4: Apply clustering using KMeans
n_clusters = 4  # Set to the number of tenses you want to classify
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(sentence_embeddings)

# Add cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Step 5: Visualize the clusters (optional)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(sentence_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_, cmap='viridis')
plt.colorbar()
plt.title("Clustered Sentences based on Tense")
plt.show()

# Step 6: Evaluate the clustering performance
silhouette_avg = silhouette_score(sentence_embeddings, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)

# Step 7: Analyze clusters without printing all sentences
print("\nClusters identified (use this to map to tenses):")
cluster_summary = df.groupby('Cluster').size()
print(cluster_summary)

# Step 8: Manually map clusters to tenses after reviewing the cluster sizes
# (Replace the values below based on your manual analysis)
cluster_to_tense = {
    0: "Past",
    1: "Present",
    2: "Future",
    3: "Present Continuous"
}

# Step 9: Add the mapped tenses to the DataFrame
df['Tense'] = df['Cluster'].map(cluster_to_tense)

# Step 10: Save the results to an Excel file
output_file = 'clustered_sentences_with_tenses.xlsx'
df.to_excel(output_file, index=False)
print(f"Tense-labeled sentences saved to {output_file}")










import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Step 1: Load the spaCy model
nlp = spacy.load("en_core_web_md")

# Step 2: Load your dataset
df = pd.read_excel('your_dataset.xlsx')  # Replace with your actual file path
sentences = df['CLEANED SENTENCE'].tolist()

# Step 3: Function to get sentence embeddings by averaging word vectors
def get_sentence_embedding(sentence, nlp_model):
    doc = nlp_model(sentence)
    word_vectors = [token.vector for token in doc if token.has_vector]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average of word vectors
    else:
        return np.zeros(nlp_model.vector_size)  # If no valid word vectors, return zero vector

# Generate embeddings for all sentences
sentence_embeddings = [get_sentence_embedding(sentence, nlp) for sentence in sentences]

# Step 4: Apply clustering using KMeans
n_clusters = 4  # Set to the number of tenses
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(sentence_embeddings)

# Add cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Step 5: Visualize the clusters (optional)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(sentence_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_, cmap='viridis')
plt.colorbar()
plt.title("Clustered Sentences based on Tense")
plt.show()

# Step 6: Evaluate clustering performance
silhouette_avg = silhouette_score(sentence_embeddings, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)

# Step 7: Analyze clusters
print("\nClusters identified (use this to map to tenses):")
cluster_summary = df.groupby('Cluster').size()
print(cluster_summary)

# Step 8: Map clusters to tenses (Update mapping based on inspection)
cluster_to_tense = {
    0: "Past",
    1: "Present",
    2: "Future",
    3: "Present Continuous"
}

# Add the mapped tenses to the DataFrame
df['Tense'] = df['Cluster'].map(cluster_to_tense)

# Step 9: Save the clustering results to an Excel file
output_file = 'clustered_sentences_with_tenses.xlsx'
df.to_excel(output_file, index=False)
print(f"Tense-labeled sentences saved to {output_file}")

# Step 10: Prepare data for classification model
X = np.array(sentence_embeddings)
y = df['Tense']

# Encode target labels into numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Step 11: Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 12: Make predictions
y_pred = clf.predict(X_test)

# Step 13: Evaluate the classification model
accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save classification results to Excel
df_test = pd.DataFrame({
    "Sentence": df.iloc[y_test.index]['CLEANED SENTENCE'].values,
    "Actual Tense": le.inverse_transform(y_test),
    "Predicted Tense": le.inverse_transform(y_pred)
})
df_test.to_excel('classification_results.xlsx', index=False)
print("Classification results saved to classification_results.xlsx")
