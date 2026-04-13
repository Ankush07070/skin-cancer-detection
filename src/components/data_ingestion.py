import pandas as pd
import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

@dataclass

class DataIngestion:
    try:
        def __init__(self, csv_path, image_dir):
            self.csv_path = csv_path
            self.image_dir = image_dir

        def load_data(self):
            df = pd.read_csv(self.csv_path)
            return df

        def add_image_paths(self, df):
            df['image_path'] = df['image_id'].apply(
            lambda x: os.path.join(self.image_dir, x + ".jpg")
            )
            return df
    except Exception as e:
     logging.error(f"Error in DataIngestion: {str(e)}")