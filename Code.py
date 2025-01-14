import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Step 1: Load dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
sheet_name = "Sheet1"  # Replace with the sheet name if needed
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Assuming your dataset has a column 'CLEANED_SENTENCE'
sentences = data['CLEANED_SENTENCE']

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)

# Step 3: Clustering
n_clusters = 4  # Adjust based on the number of tenses you expect
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
data['Cluster'] = clusters

# Step 4: Analyze and Map Clusters to Tenses
# Inspect sentences in each cluster to map them to tenses
cluster_samples = {}
for cluster_id in range(n_clusters):
    cluster_samples[cluster_id] = data[data['Cluster'] == cluster_id]['CLEANED_SENTENCE'].head(5).tolist()
    print(f"\nCluster {cluster_id} Samples:")
    for sentence in cluster_samples[cluster_id]:
        print(f"  - {sentence}")

# After inspection, map clusters to tenses (e.g., Present, Past, Future, Present Continuous)
cluster_to_tense = {
    0: 'Present', 
    1: 'Past', 
    2: 'Future', 
    3: 'Present Continuous'
}  # Adjust based on manual inspection

data['Predicted_Tense'] = data['Cluster'].map(cluster_to_tense)

# Step 5: Cluster Size
print("\nCluster Sizes:")
print(data['Cluster'].value_counts())

# Step 6: Interactive Input for Dynamic Classification
print("\nEnter a sentence to classify:")
user_input = input()
user_tfidf = vectorizer.transform([user_input])
user_cluster = kmeans.predict(user_tfidf)[0]
user_tense = cluster_to_tense.get(user_cluster, "Unknown")

print(f"\nPredicted Tense for '{user_input}': {user_tense}")

# Step 7: Save Results
output_path = "classified_dataset.xlsx"
data.to_excel(output_path, index=False)
print(f"\nClassified dataset saved to {output_path}")
