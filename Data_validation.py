import pandas as pd

class DataValidation:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def validate_data(self):
        """Validate if data meets the required schema."""
        if not all(col in self.data.columns for col in ['sentence', 'label']):
            raise ValueError("Data is missing required columns: 'sentence' and 'label'")

        if self.data.isnull().sum().sum() > 0:
            raise ValueError("Data contains missing values.")
        
        print("Data validation passed.")
        return self.data
        
