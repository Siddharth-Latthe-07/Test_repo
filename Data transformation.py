import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class DataTransformation:
    def __init__(self, config):
        self.vectorizer_output_path = config["data_transformation"]["vectorizer_output_path"]

    def transform_data(self, sentences):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        joblib.dump(vectorizer, self.vectorizer_output_path)
        return X

    def run(self, train_data, test_data):
        X_train = self.transform_data(train_data["sentence"])
        X_test = self.transform_data(test_data["sentence"])
        return X_train, X_test
      
