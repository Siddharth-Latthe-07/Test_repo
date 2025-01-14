from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# Step 1: Load the data
data = pd.DataFrame({'CLEANED_SENTENCE': [
    "I am reading a book.", "I read a book.", 
    "I will read a book.", "I am going to read a book.",
    "She was writing a letter.", "They will complete the project tomorrow.",
    "He is eating lunch.", "We studied all night."
]})

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['CLEANED_SENTENCE'])

# Step 3: Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
data['Cluster'] = clusters

# Step 4: Map clusters to tenses (manual inspection required)
# After inspecting some sentences from each cluster, assign labels to clusters
cluster_to_tense = {0: 'Present', 1: 'Past', 2: 'Future', 3: 'Present Continuous'}
data['Tense'] = data['Cluster'].map(cluster_to_tense)

# Print results
print(data[['CLEANED_SENTENCE', 'Tense']])
