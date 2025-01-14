import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Load Dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
sheet_name = "Sheet1"  # Replace with the sheet name if needed
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Assuming your dataset has a column 'CLEANED_SENTENCE'
sentences = data['CLEANED_SENTENCE']

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)

# Step 3: Clustering (Pseudo-Labeling)
n_clusters = 4  # Number of tenses
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
data['Cluster'] = clusters

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, clusters, test_size=0.2, random_state=42)

# Step 5: Train an ML Model (Logistic Regression)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_clusters), yticklabels=range(n_clusters))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 7: Save Model and Vectorizer
joblib.dump(model, "tense_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel and vectorizer saved.")

# Step 8: Interactive Input for Dynamic Classification
print("\nEnter a sentence to classify:")
user_input = input()
user_tfidf = vectorizer.transform([user_input])
user_cluster = model.predict(user_tfidf)[0]

print(f"\nPredicted Cluster for '{user_input}': {user_cluster}")
