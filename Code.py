from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import pandas as pd

# Step 1: Load the data
data = pd.DataFrame({'CLEANED_SENTENCE': [
    "I am reading a book.", "I read a book.", 
    "I will read a book.", "I am going to read a book."
]})

# Step 2: Sentence Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight pre-trained model
sentence_embeddings = model.encode(data['CLEANED_SENTENCE'])

# Step 3: Dimensionality Reduction (optional, for visualization)
pca = PCA(n_components=2)  # Reduce dimensions to 2D for visualization
reduced_embeddings = pca.fit_transform(sentence_embeddings)

# Step 4: Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(reduced_embeddings)

# Step 5: Assign Cluster Labels
data['Cluster'] = clusters

# Map clusters to tenses (based on manual inspection of clusters)
cluster_to_tense = {0: 'Present', 1: 'Past', 2: 'Future', 3: 'Present Continuous'}
data['Tense'] = data['Cluster'].map(cluster_to_tense)

print(data[['CLEANED_SENTENCE', 'Tense']])
