from sklearn.feature_extraction.text import TfidfVectorizer

class DataTransformation:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def transform_data(self):
        """Transform text data using TF-IDF vectorization."""
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.data['sentence'])
        y = self.data['label']
        
        print("Data transformation using TF-IDF completed.")
        return X, y, vectorizer
        
