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








import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
import random

# Step 1: Load the spaCy model
nlp = spacy.load("en_core_web_md")

# Step 2: Load the dataset
df = pd.read_excel('your_dataset.xlsx')  # Replace with the actual file path
sentences = df['CLEANED SENTENCE'].tolist()

# Step 3: Generate sentence embeddings using spaCy
def get_sentence_embedding(sentence, nlp_model):
    doc = nlp_model(sentence)
    word_vectors = [token.vector for token in doc if token.has_vector]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word vectors
    else:
        return np.zeros(nlp_model.vector_size)  # Return zero vector if no valid words

sentence_embeddings = [get_sentence_embedding(sentence, nlp) for sentence in sentences]

# Step 4: Perform clustering using KMeans
n_clusters = 4  # Assuming 4 tenses
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(sentence_embeddings)

# Add cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Step 5: Map clusters to tenses (manual mapping after analyzing clusters)
cluster_to_tense = {
    0: "Past",
    1: "Present",
    2: "Future",
    3: "Present Continuous"
}
df['Tense'] = df['Cluster'].map(cluster_to_tense)

# Save clustering results
df.to_excel('clustered_sentences_with_tenses.xlsx', index=False)

# Step 6: Data Augmentation using Synonym Replacement
def augment_sentence(sentence, nlp_model, num_replacements=2):
    doc = nlp_model(sentence)
    augmented_sentence = []
    for token in doc:
        if token.is_alpha and random.random() < 0.3:  # Replace only some words randomly
            synonyms = wordnet.synsets(token.text)
            if synonyms:
                replacement = synonyms[0].lemmas()[0].name()
                augmented_sentence.append(replacement if replacement else token.text)
            else:
                augmented_sentence.append(token.text)
        else:
            augmented_sentence.append(token.text)
    return ' '.join(augmented_sentence)

# Generate augmented data
augmented_sentences = []
augmented_tenses = []
for idx, row in df.iterrows():
    for _ in range(3):  # Generate 3 augmented sentences for each original
        augmented_sentences.append(augment_sentence(row['CLEANED SENTENCE'], nlp))
        augmented_tenses.append(row['Tense'])

# Add augmented data to the original dataset
augmented_embeddings = [get_sentence_embedding(sentence, nlp) for sentence in augmented_sentences]
X = np.vstack((sentence_embeddings, augmented_embeddings))
y = df['Tense'].tolist() + augmented_tenses

# Combine the original and augmented data into a new DataFrame
augmented_df = pd.DataFrame({
    'Sentence': sentences + augmented_sentences,
    'Tense': df['Tense'].tolist() + augmented_tenses
})
augmented_df.to_excel('augmented_dataset_with_tenses.xlsx', index=False)

# Step 7: Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 8: Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Step 9: Perform cross-validation
clf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Step 10: Train the model on the entire training data
clf.fit(X_train, y_train)

# Step 11: Make predictions and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Set Accuracy:", accuracy)
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

# Save classification results
df_test = pd.DataFrame({
    "Sentence": augmented_df.iloc[y_test.index]['Sentence'].values,
    "Actual Tense": le.inverse_transform(y_test),
    "Predicted Tense": le.inverse_transform(y_pred)
})
df_test.to_excel('classification_results_with_augmentation.xlsx', index=False)
print("Classification results saved to classification_results_with_augmentation.xlsx")





import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
import random

# Step 1: Load the spaCy model
nlp = spacy.load("en_core_web_md")

# Step 2: Load the dataset
df = pd.read_excel('your_dataset.xlsx')  # Replace with the actual file path
sentences = df['CLEANED SENTENCE'].tolist()

# Step 3: Generate sentence embeddings using spaCy
def get_sentence_embedding(sentence, nlp_model):
    doc = nlp_model(sentence)
    word_vectors = [token.vector for token in doc if token.has_vector]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word vectors
    else:
        return np.zeros(nlp_model.vector_size)  # Return zero vector if no valid words

sentence_embeddings = [get_sentence_embedding(sentence, nlp) for sentence in sentences]

# Step 4: Perform clustering using KMeans
n_clusters = 4  # Assuming 4 tenses
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(sentence_embeddings)

# Add cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Step 5: Map clusters to tenses (manual mapping after analyzing clusters)
cluster_to_tense = {
    0: "Past",
    1: "Present",
    2: "Future",
    3: "Present Continuous"
}
df['Tense'] = df['Cluster'].map(cluster_to_tense)

# Save clustering results
df.to_excel('clustered_sentences_with_tenses.xlsx', index=False)

# Step 6: Visualize Clusters using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(sentence_embeddings)

plt.figure(figsize=(10, 8))
for cluster in range(n_clusters):
    cluster_points = reduced_embeddings[kmeans.labels_ == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster} ({cluster_to_tense[cluster]})")
plt.title("Cluster Visualization (Tenses)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.savefig("cluster_visualization.png")  # Save the cluster visualization
plt.show()

# Step 7: Data Augmentation using Synonym Replacement
def augment_sentence(sentence, nlp_model, num_replacements=2):
    doc = nlp_model(sentence)
    augmented_sentence = []
    for token in doc:
        if token.is_alpha and random.random() < 0.3:  # Replace only some words randomly
            synonyms = wordnet.synsets(token.text)
            if synonyms:
                replacement = synonyms[0].lemmas()[0].name()
                augmented_sentence.append(replacement if replacement else token.text)
            else:
                augmented_sentence.append(token.text)
        else:
            augmented_sentence.append(token.text)
    return ' '.join(augmented_sentence)

# Generate augmented data
augmented_sentences = []
augmented_tenses = []
for idx, row in df.iterrows():
    for _ in range(3):  # Generate 3 augmented sentences for each original
        augmented_sentences.append(augment_sentence(row['CLEANED SENTENCE'], nlp))
        augmented_tenses.append(row['Tense'])

# Add augmented data to the original dataset
augmented_embeddings = [get_sentence_embedding(sentence, nlp) for sentence in augmented_sentences]
X = np.vstack((sentence_embeddings, augmented_embeddings))
y = df['Tense'].tolist() + augmented_tenses

# Combine the original and augmented data into a new DataFrame
augmented_df = pd.DataFrame({
    'Sentence': sentences + augmented_sentences,
    'Tense': df['Tense'].tolist() + augmented_tenses
})
augmented_df.to_excel('augmented_dataset_with_tenses.xlsx', index=False)

# Step 8: Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 9: Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Step 10: Perform cross-validation
clf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Step 11: Train the model on the entire training data
clf.fit(X_train, y_train)

# Step 12: Make predictions and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Set Accuracy:", accuracy)
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

# Save classification results
df_test = pd.DataFrame({
    "Sentence": X_test[:, 0] if len(X_test) > 0 else [],
    "Actual Tense": le.inverse_transform(y_test),
    "Predicted Tense": le.inverse_transform(y_pred)
})
df_test.to_excel('classification_results_with_augmentation.xlsx', index=False)
print("Classification results saved to classification_results_with_augmentation.xlsx")






import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load spaCy model
nlp = spacy.load("en_core_web_md")

# Step 2: Load Dataset
df = pd.read_excel('your_dataset.xlsx')  # Replace with your file
sentences = df['CLEANED SENTENCE'].tolist()

# Step 3: Sentence Embeddings
def get_sentence_embedding(sentence, nlp_model):
    doc = nlp_model(sentence)
    word_vectors = [token.vector for token in doc if token.has_vector]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word vectors
    else:
        return np.zeros(nlp_model.vector_size)  # Return zero vector if no valid words

sentence_embeddings = np.array([get_sentence_embedding(sentence, nlp) for sentence in sentences])

# Step 4: Dimensionality Reduction with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
reduced_embeddings = tsne.fit_transform(sentence_embeddings)

# Step 5: KMeans Clustering
n_clusters = 4  # Assuming 4 tenses
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(reduced_embeddings)

# Add cluster labels to DataFrame
df['Cluster'] = kmeans.labels_

# Manual Mapping of Clusters to Tenses
cluster_to_tense = {
    0: "Past",
    1: "Present",
    2: "Future",
    3: "Present Continuous"
}
df['Tense'] = df['Cluster'].map(cluster_to_tense)

# Save the clustered data
df.to_excel('clustered_sentences_with_tenses.xlsx', index=False)

# Step 6: Visualize Clusters and Boundaries
plt.figure(figsize=(10, 8))
x_min, x_max = reduced_embeddings[:, 0].min() - 1, reduced_embeddings[:, 0].max() + 1
y_min, y_max = reduced_embeddings[:, 1].min() - 1, reduced_embeddings[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_clusters = kmeans.predict(grid_points)

plt.contourf(xx, yy, grid_clusters.reshape(xx.shape), alpha=0.2, cmap="viridis")
for cluster in range(n_clusters):
    cluster_points = reduced_embeddings[kmeans.labels_ == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster} ({cluster_to_tense[cluster]})")
plt.title("Cluster Visualization with Boundaries (Tenses)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.savefig("cluster_visualization_with_boundaries.png")  # Save the plot
plt.show()

# Step 7: Build a Classification Model
le = LabelEncoder()
y = le.fit_transform(df['Tense'])
X = np.vstack((reduced_embeddings, reduced_embeddings))  # Use reduced embeddings for classification

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cross-Validation
clf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Train on Full Training Set
clf.fit(X_train, y_train)

# Evaluate Model
y_pred = clf.predict(X_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save Results
df_test = pd.DataFrame({
    "Sentence": df.iloc[X_test][:, 0],
    "Actual Tense": le.inverse_transform(y_test),
    "Predicted Tense": le.inverse_transform(y_pred)
})
df_test.to_excel('classification_results.xlsx', index=False)

print("Results saved to 'classification_results.xlsx'")










import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load spaCy model
nlp = spacy.load("en_core_web_md")

# Step 2: Load Dataset
df = pd.read_excel('your_dataset.xlsx')  # Replace with your file
sentences = df['CLEANED SENTENCE'].tolist()

# Step 3: Sentence Embeddings
def get_sentence_embedding(sentence, nlp_model):
    doc = nlp_model(sentence)
    word_vectors = [token.vector for token in doc if token.has_vector]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word vectors
    else:
        return np.zeros(nlp_model.vector_size)  # Return zero vector if no valid words

sentence_embeddings = np.array([get_sentence_embedding(sentence, nlp) for sentence in sentences], dtype=np.float64)

# Ensure embeddings are not all zeros (fallback mechanism)
valid_embeddings = np.any(sentence_embeddings, axis=1)
if not valid_embeddings.all():
    print("Some embeddings are invalid and will be removed.")
    df = df[valid_embeddings]
    sentence_embeddings = sentence_embeddings[valid_embeddings]

# Step 4: Dimensionality Reduction with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
reduced_embeddings = tsne.fit_transform(sentence_embeddings.astype(np.float32))  # Ensure float32 precision

# Step 5: KMeans Clustering
n_clusters = 4  # Assuming 4 tenses
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(reduced_embeddings)

# Add cluster labels to DataFrame
df['Cluster'] = kmeans.labels_

# Manual Mapping of Clusters to Tenses
cluster_to_tense = {
    0: "Past",
    1: "Present",
    2: "Future",
    3: "Present Continuous"
}
df['Tense'] = df['Cluster'].map(cluster_to_tense)

# Save the clustered data
df.to_excel('clustered_sentences_with_tenses.xlsx', index=False)

# Step 6: Visualize Clusters and Boundaries
plt.figure(figsize=(10, 8))
x_min, x_max = reduced_embeddings[:, 0].min() - 1, reduced_embeddings[:, 0].max() + 1
y_min, y_max = reduced_embeddings[:, 1].min() - 1, reduced_embeddings[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_clusters = kmeans.predict(grid_points)

plt.contourf(xx, yy, grid_clusters.reshape(xx.shape), alpha=0.2, cmap="viridis")
for cluster in range(n_clusters):
    cluster_points = reduced_embeddings[kmeans.labels_ == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster} ({cluster_to_tense[cluster]})")
plt.title("Cluster Visualization with Boundaries (Tenses)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.savefig("cluster_visualization_with_boundaries.png")  # Save the plot
plt.show()

# Step 7: Build a Classification Model
le = LabelEncoder()
y = le.fit_transform(df['Tense'])
X = reduced_embeddings  # Use reduced embeddings for classification

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cross-Validation
clf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Train on Full Training Set
clf.fit(X_train, y_train)

# Evaluate Model
y_pred = clf.predict(X_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save Results
df_test = pd.DataFrame({
    "Sentence": df.iloc[X_test.index]['CLEANED SENTENCE'].values,
    "Actual Tense": le.inverse_transform(y_test),
    "Predicted Tense": le.inverse_transform(y_pred)
})
df_test.to_excel('classification_results.xlsx', index=False)

print("Results saved to 'classification_results.xlsx'")






import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load spaCy model
nlp = spacy.load("en_core_web_md")

# Step 2: Load Dataset
df = pd.read_excel('your_dataset.xlsx')  # Replace with your file
sentences = df['CLEANED SENTENCE'].tolist()

# Step 3: Sentence Embeddings
def get_sentence_embedding(sentence, nlp_model):
    doc = nlp_model(sentence)
    word_vectors = [token.vector for token in doc if token.has_vector]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word vectors
    else:
        return np.zeros(nlp_model.vector_size)  # Return zero vector if no valid words

sentence_embeddings = np.array([get_sentence_embedding(sentence, nlp) for sentence in sentences], dtype=np.float32)  # Use float32

# Ensure embeddings are not all zeros (fallback mechanism)
valid_embeddings = np.any(sentence_embeddings, axis=1)
if not valid_embeddings.all():
    print("Some embeddings are invalid and will be removed.")
    df = df[valid_embeddings]
    sentence_embeddings = sentence_embeddings[valid_embeddings]

# Step 4: Dimensionality Reduction with t-SNE
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    reduced_embeddings = tsne.fit_transform(sentence_embeddings)  # Ensure compatibility
except ValueError as e:
    print(f"Error during t-SNE: {e}")
    raise

# Step 5: KMeans Clustering
n_clusters = 4  # Assuming 4 tenses
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(reduced_embeddings)

# Add cluster labels to DataFrame
df['Cluster'] = kmeans.labels_

# Manual Mapping of Clusters to Tenses
cluster_to_tense = {
    0: "Past",
    1: "Present",
    2: "Future",
    3: "Present Continuous"
}
df['Tense'] = df['Cluster'].map(cluster_to_tense)

# Save the clustered data
df.to_excel('clustered_sentences_with_tenses.xlsx', index=False)

# Step 6: Visualize Clusters and Boundaries
plt.figure(figsize=(10, 8))
x_min, x_max = reduced_embeddings[:, 0].min() - 1, reduced_embeddings[:, 0].max() + 1
y_min, y_max = reduced_embeddings[:, 1].min() - 1, reduced_embeddings[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_clusters = kmeans.predict(grid_points)

plt.contourf(xx, yy, grid_clusters.reshape(xx.shape), alpha=0.2, cmap="viridis")
for cluster in range(n_clusters):
    cluster_points = reduced_embeddings[kmeans.labels_ == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster} ({cluster_to_tense[cluster]})")
plt.title("Cluster Visualization with Boundaries (Tenses)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.savefig("cluster_visualization_with_boundaries.png")  # Save the plot
plt.show()

# Step 7: Build a Classification Model
le = LabelEncoder()
y = le.fit_transform(df['Tense'])
X = reduced_embeddings  # Use reduced embeddings for classification

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cross-Validation
clf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Train on Full Training Set
clf.fit(X_train, y_train)

# Evaluate Model
y_pred = clf.predict(X_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save Results
df_test = pd.DataFrame({
    "Sentence": df.iloc[X_test.index]['CLEANED SENTENCE'].values,
    "Actual Tense": le.inverse_transform(y_test),
    "Predicted Tense": le.inverse_transform(y_pred)
})
df_test.to_excel('classification_results.xlsx', index=False)

print("Results saved to 'classification_results.xlsx'")









import pandas as pd
import spacy
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load spaCy model
nlp = spacy.load("en_core_web_md")

# Step 2: Load Dataset
df = pd.read_excel('your_dataset.xlsx')  # Replace with your file
sentences = df['CLEANED SENTENCE'].tolist()

# Step 3: Sentence Embeddings
def get_sentence_embedding(sentence, nlp_model):
    doc = nlp_model(sentence)
    word_vectors = [token.vector for token in doc if token.has_vector]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word vectors
    else:
        return np.zeros(nlp_model.vector_size)  # Return zero vector if no valid words

sentence_embeddings = np.array([get_sentence_embedding(sentence, nlp) for sentence in sentences], dtype=np.float32)  # Use float32

# Ensure embeddings are not all zeros (fallback mechanism)
valid_embeddings = np.any(sentence_embeddings, axis=1)
if not valid_embeddings.all():
    print("Some embeddings are invalid and will be removed.")
    df = df[valid_embeddings]
    sentence_embeddings = sentence_embeddings[valid_embeddings]

# Step 4: Dimensionality Reduction with UMAP
try:
    umap = UMAP(n_components=2, random_state=42, metric="cosine")
    reduced_embeddings = umap.fit_transform(sentence_embeddings)  # Ensure compatibility
except ValueError as e:
    print(f"Error during UMAP: {e}")
    raise

# Step 5: KMeans Clustering
n_clusters = 4  # Assuming 4 tenses
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(reduced_embeddings)

# Add cluster labels to DataFrame
df['Cluster'] = kmeans.labels_

# Manual Mapping of Clusters to Tenses
cluster_to_tense = {
    0: "Past",
    1: "Present",
    2: "Future",
    3: "Present Continuous"
}
df['Tense'] = df['Cluster'].map(cluster_to_tense)

# Save the clustered data
df.to_excel('clustered_sentences_with_tenses.xlsx', index=False)

# Step 6: Visualize Clusters
plt.figure(figsize=(10, 8))
for cluster in range(n_clusters):
    cluster_points = reduced_embeddings[kmeans.labels_ == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster} ({cluster_to_tense[cluster]})")
plt.title("UMAP-Based Clusters (Tenses)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend()
plt.savefig("umap_cluster_visualization.png")  # Save the plot
plt.show()

# Step 7: Build a Classification Model
le = LabelEncoder()
y = le.fit_transform(df['Tense'])
X = reduced_embeddings  # Use reduced embeddings for classification

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cross-Validation
clf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Train on Full Training Set
clf.fit(X_train, y_train)

# Evaluate Model
y_pred = clf.predict(X_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save Results
df_test = pd.DataFrame({
    "Sentence": df.iloc[X_test.index]['CLEANED SENTENCE'].values,
    "Actual Tense": le.inverse_transform(y_test),
    "Predicted Tense": le.inverse_transform(y_pred)
})
df_test.to_excel('classification_results.xlsx', index=False)

print("Results saved to 'classification_results.xlsx'")





import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load SpaCy model
nlp = spacy.load("en_core_web_sm")  # Use a smaller model with dependency parsing

# Step 2: Load Dataset
df = pd.read_excel("your_dataset.xlsx")  # Replace with your dataset file
sentences = df["CLEANED SENTENCE"].tolist()

# Step 3: Extract Syntactic Features
def extract_syntactic_features(sentence, nlp_model):
    doc = nlp_model(sentence)
    features = []
    for token in doc:
        features.append([
            token.dep_,           # Dependency relation
            token.tag_,           # Detailed POS tag
            token.head.text,      # Head word of the dependency
            token.head.pos_       # POS tag of the head word
        ])
    return features

# Convert syntactic features into numerical vectors
def sentence_to_features(sentence, nlp_model):
    doc = nlp_model(sentence)
    features = []
    for token in doc:
        # Vectorize syntactic information
        features.append([
            token.vector,         # Word vector
            len(token.dep_),      # Dependency length
            len(token.tag_),      # Tag length
            len(token.head.text)  # Head word length
        ])
    return np.mean(features, axis=0) if features else np.zeros(nlp_model.vector_size)

syntactic_embeddings = np.array([sentence_to_features(sentence, nlp) for sentence in sentences])

# Step 4: Dimensionality Reduction for Clustering
pca = PCA(n_components=2, random_state=42)
reduced_embeddings = pca.fit_transform(syntactic_embeddings)

# Step 5: KMeans Clustering
n_clusters = 4  # Assuming 4 tenses: Past, Present, Future, Present Continuous
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(reduced_embeddings)
df["Cluster"] = kmeans.labels_

# Map clusters to tenses manually
cluster_to_tense = {
    0: "Past",
    1: "Present",
    2: "Future",
    3: "Present Continuous"
}
df["Tense"] = df["Cluster"].map(cluster_to_tense)

# Save clustered data
df.to_excel("syntactic_clustered_sentences.xlsx", index=False)

# Step 6: Visualize Clusters
plt.figure(figsize=(10, 8))
for cluster in range(n_clusters):
    cluster_points = reduced_embeddings[kmeans.labels_ == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster} ({cluster_to_tense[cluster]})")
plt.title("PCA-Based Clusters (Syntactic Features)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.savefig("syntactic_clusters.png")
plt.show()

# Step 7: Build Classification Model
le = LabelEncoder()
y = le.fit_transform(df["Tense"])
X = syntactic_embeddings  # Use syntactic embeddings for classification

# Split Dataset
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y
)

# Cross-Validation
clf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Train on Full Training Set
clf.fit(X_train, y_train)

# Evaluate Model
y_pred = clf.predict(X_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save Results
df_test = pd.DataFrame({
    "Sentence": df.iloc[test_indices]["CLEANED SENTENCE"].values,
    "Actual Tense": le.inverse_transform(y_test),
    "Predicted Tense": le.inverse_transform(y_pred)
})
df_test.to_excel("syntactic_classification_results.xlsx", index=False)

print("Results saved to 'syntactic_classification_results.xlsx'")








import pandas as pd
import numpy as np
import spacy
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_excel("input_dataset.xlsx")  # Replace with your actual dataset
sentences = df['CLEANED SENTENCE'].tolist()

# Feature extraction function (fixed-length vectors)
def extract_features(sentence):
    doc = nlp(sentence)
    # Extracting syntactic dependency embeddings
    syntactic_features = np.array([token.dep for token in doc if token.dep])
    # Use mean of word vectors for fixed length
    sentence_vector = doc.vector
    return np.concatenate([sentence_vector, syntactic_features], axis=None)

# Extract features for all sentences
X = []
for sentence in sentences:
    try:
        features = extract_features(sentence)
        X.append(features)
    except Exception as e:
        print(f"Error processing sentence: {sentence}, Error: {e}")

# Convert to a numpy array
X = np.array(X)

# Handle any issues with shape (e.g., pad sequences if necessary)
max_length = max(len(x) for x in X)
X = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in X])

# Clustering based on tense
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Add cluster labels to the dataframe
df['TENSE_CLUSTER'] = labels

# Map clusters to tenses (manual mapping based on inspection)
tense_mapping = {0: 'Past', 1: 'Present', 2: 'Future', 3: 'Present Continuous'}
df['TENSE'] = df['TENSE_CLUSTER'].map(tense_mapping)

# Save the new dataset
df.to_excel("output_dataset_with_tense.xlsx", index=False)

# Train a classification model
le = LabelEncoder()
y = le.fit_transform(df['TENSE'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Cluster Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()




import pandas as pd
import numpy as np
import spacy
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_excel("input_dataset.xlsx")  # Replace with your dataset file
sentences = df['CLEANED SENTENCE'].tolist()

# Feature extraction function
def extract_features(sentence):
    doc = nlp(sentence)
    # Use fixed-length sentence embedding from SpaCy
    sentence_vector = doc.vector
    return sentence_vector

# Extract features for all sentences
X = []
for sentence in sentences:
    try:
        features = extract_features(sentence)
        X.append(features)
    except Exception as e:
        print(f"Error processing sentence: {sentence}, Error: {e}")

# Convert to a numpy array (ensure consistent shape)
X = np.array(X)

# Clustering based on tense
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Add cluster labels to the dataframe
df['TENSE_CLUSTER'] = labels

# Map clusters to tenses (manual mapping based on inspection)
tense_mapping = {0: 'Past', 1: 'Present', 2: 'Future', 3: 'Present Continuous'}
df['TENSE'] = df['TENSE_CLUSTER'].map(tense_mapping)

# Save the new dataset
df.to_excel("output_dataset_with_tense.xlsx", index=False)

# Train a classification model
le = LabelEncoder()
y = le.fit_transform(df['TENSE'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Cluster Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
