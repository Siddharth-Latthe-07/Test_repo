# Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
file_path = "your_dataset.xlsx"  # Replace with your file path
data = pd.read_excel(file_path)
data = data[['cleaned_sentence']]  # Ensure the dataset has the correct column
data['LABEL'] = None  # Add a placeholder for supervised learning labels

# Placeholder Labels (Manually Assign Initial Labels if Available)
# Replace with actual labels or assign labels programmatically
data.loc[:50, 'LABEL'] = 'Present'
data.loc[51:100, 'LABEL'] = 'Past'
data.loc[101:150, 'LABEL'] = 'Future'
data.loc[151:, 'LABEL'] = 'Present Continuous'

# Step 1: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_sentence'])
y = data['LABEL']

# Step 2: Supervised Learning Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
print("\nClassification Report (Supervised Learning):")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 3: Clustering for Verification
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)
data['Cluster'] = clusters

# Step 4: Compare Clustering with True Labels
comparison = pd.crosstab(data['LABEL'], data['Cluster'], rownames=['True Label'], colnames=['Cluster'])
print("\nCluster vs True Label Comparison:")
print(comparison)
