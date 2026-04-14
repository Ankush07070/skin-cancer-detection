from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from src.logger import logging
from src.exception import CustomException
import sys

class SkinCancerDataset(Dataset):
    try:
            def __init__(self, df, transform=None):
                self.df = df
                self.transform = transform

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]

                img_path = row['image_path']
                label = row['label']

                image = Image.open(img_path).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                return image, torch.tensor(label, dtype=torch.long)
    except Exception as e:
        logging.error(f"Error in SkinCancerDataset: {str(e)}")
        raise CustomException(e, sys)


def create_dataloader(df, transform, batch_size=32, shuffle=True):
    try:
        dataset = SkinCancerDataset(df, transform=transform)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2
        )

        return loader
    except Exception as e:
        logging.error(f"Error in create_dataloader: {str(e)}")
        raise CustomException(e, sys)