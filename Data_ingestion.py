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
      
