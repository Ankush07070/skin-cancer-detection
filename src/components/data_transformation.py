import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import sys
@dataclass

class DataTransformation:
    try:
            def encode_labels(self, df):
                label_map = {
                    'nv': 0, 'mel': 1, 'bkl': 2,
                    'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6
                }
                df['label'] = df['dx'].map(label_map)
                return df, label_map

            def balance_data(self, df):
                max_size = df['label'].value_counts().max()
                df_list = []

                for label in df['label'].unique():
                    df_class = df[df['label'] == label]
                    df_resampled = resample(df_class, replace=True,
                                            n_samples=max_size, random_state=42)
                    df_list.append(df_resampled)

                return pd.concat(df_list).sample(frac=1)

            def split_data(self, df):
                return train_test_split(
                    df, test_size=0.2,
                    stratify=df['label'], random_state=42
                )

            def get_class_weights(self, df):
                weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(df['label']),
                    y=df['label']
                )
                return dict(enumerate(weights))
    except Exception as e:
        logging.error(f"Error in DataTransformation: {str(e)}")
        raise CustomException(e, sys)