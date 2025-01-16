from src.components.data_validation import DataValidation

class DataValidationPipeline:
    def __init__(self, data):
        self.data = data
        self.data_validation = DataValidation(data)

    def run(self):
        validated_data = self.data_validation.validate_data()
        return validated_data
        
