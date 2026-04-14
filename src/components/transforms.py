import sys

from torchvision import transforms
from src.logger import logging
from src.exception import CustomException
def get_train_transforms():
    try:
            return transforms.Compose([
                transforms.Resize((224, 224)),

                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),

                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    except Exception as e:
        logging.error(f"Error in get_train_transforms: {str(e)}")
        raise CustomException(e, sys)

def get_val_transforms():
    try:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    except Exception as e:
        logging.error(f"Error in get_val_transforms: {str(e)}")
        raise CustomException(e, sys)