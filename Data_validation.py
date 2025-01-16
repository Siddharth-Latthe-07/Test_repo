import yaml
import pandas as pd

class DataValidation:
    def __init__(self, schema_file_path):
        self.schema_file_path = schema_file_path

    def validate_schema(self, data):
        with open(self.schema_file_path, "r") as f:
            schema = yaml.safe_load(f)

        required_columns = [col["name"] for col in schema["columns"]]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        return True

    def run(self, data):
        self.validate_schema(data)
        print("Data validation successful.")
      
