import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, config):
        self.file_path = config["data_ingestion"]["file_path"]
        self.output_dir = config["data_ingestion"]["output_dir"]
        self.test_size = config["data_ingestion"]["test_size"]
        self.random_state = config["data_ingestion"]["random_state"]

    def load_data(self):
        data = pd.read_excel(self.file_path)
        return data

    def split_data(self, data):
        train, test = train_test_split(
            data, test_size=self.test_size, random_state=self.random_state
        )
        return train, test

    def save_data(self, train, test):
        os.makedirs(self.output_dir, exist_ok=True)
        train.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)

    def run(self):
        data = self.load_data()
        train, test = self.split_data(data)
        self.save_data(train, test)
        return train, test










import pandas as pd
import os
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.raw_data_dir = self.config.raw_data_dir
        self.file_name = self.config.file_name
        self.data_file_path = os.path.join(self.raw_data_dir, self.file_name)
        
    def check_data_existence(self):
        """Check if the data file exists at the given path."""
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"The data file {self.file_name} does not exist at the specified path: {self.raw_data_dir}")
        print(f"Data file found at {self.data_file_path}")

    def ingest_data(self):
        """Ingest the data from the source (Excel file) and save it as a pandas DataFrame."""
        self.check_data_existence()

        try:
            # Load the data from the Excel file into a pandas DataFrame
            data = pd.read_excel(self.data_file_path)

            # Validate that necessary columns exist
            required_columns = ['sentence', 'label']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"The data file is missing required columns: {', '.join(required_columns)}")
            
            print("Data ingestion successful.")
            # Return the data for further processing
            return data

        except Exception as e:
            print(f"Error while ingesting data: {e}")
            raise e
            
