import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
@dataclass
class DataValidation:
    try:
       def remove_missing_images(self, df):
        df = df[df['image_path'].apply(os.path.exists)]
        return df
    except Exception as e:
        raise CustomException(str(e), sys)