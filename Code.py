from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np

# Step 1: Load the data
data = pd.DataFrame({'CLEANED_SENTENCE': [
    "I am reading a book.", "I read a book.", 
    "I will read a book.", "I am going to read a book."
]})

# Step 2: Pre-trained Word Embeddings
# Load pre-trained word embeddings (e.g., GloVe or Word2Vec)
# For this example, let's assume we have a GloVe model in text format
# Download GloVe (e.g., glove.6B.50d.txt) and replace the path with the correct one
word_vectors = KeyedVectors.load_word2vec_format("glove.6B.50d.txt", binary=False, no_header=True)

# Step 3: Define a function to get sentence embeddings
def sentence_to_vector(sentence, model, embedding_dim=50):
    words = sentence.lower().split()  # Tokenize sentence
    vectors = [model[word] for word in words if word in model]
    if len(vectors) == 0:
        return np.zeros(embedding_dim)  # Handle sentences with no known words
    return np.mean(vectors, axis=0)  # Average word embeddings

# Step 4: Compute sentence embeddings
embedding_dim = word_vectors.vector_size  # Get the dimension of word vectors
data['Sentence_Embedding'] = data['CLEANED_SENTENCE'].apply(
    lambda x: sentence_to_vector(x, word_vectors, embedding_dim))

# Step 5: Clustering
X = np.vstack(data['Sentence_Embedding'].to_numpy())  # Stack embeddings into a matrix
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Step 6: Map clusters to tenses (manual inspection required)
cluster_to_tense = {0: 'Present', 1: 'Past', 2: 'Future', 3: 'Present Continuous'}
data['Tense'] = data['Cluster'].map(cluster_to_tense)

print(data[['CLEANED_SENTENCE', 'Tense']])
