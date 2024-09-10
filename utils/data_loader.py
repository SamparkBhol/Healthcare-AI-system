import os
import pandas as pd
import numpy as np
import json

class DataLoader:
    def __init__(self, data_dir, file_format="csv"):
        self.data_dir = data_dir
        self.file_format = file_format.lower()

    def load_data(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)

        if self.file_format == "csv":
            data = pd.read_csv(file_path)
        elif self.file_format == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
        elif self.file_format == "npz":
            data = np.load(file_path, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

        return data

    def split_data(self, data, train_ratio=0.8):
        # Split data into training and testing sets
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        return train_data, test_data

    def preprocess_data(self, data, normalize=True):
        # Basic preprocessing steps for the dataset
        if normalize:
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

if __name__ == "__main__":
    loader = DataLoader(data_dir="path_to_healthcare_data", file_format="csv")
    data = loader.load_data(file_name="patient_records.csv")
    train_data, test_data = loader.split_data(data)
    processed_data = loader.preprocess_data(train_data)
