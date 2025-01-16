from src.components.data_transformation import DataTransformation

class DataTransformationPipeline:
    def __init__(self, data):
        self.data = data
        self.data_transformation = DataTransformation(data)

    def run(self):
        X, y, vectorizer = self.data_transformation.transform_data()
        return X, y, vectorizer
        
