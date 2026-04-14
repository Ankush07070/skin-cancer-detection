import os
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
import sys
from src.exception import CustomException   
from src.logger import logging
import pandas as pd
from src.components.data_loader import create_dataloader
from src.components.transforms import get_train_transforms, get_val_transforms
import torch
from src.components.model import SkinCancerModel
from src.components.trainer import train_one_epoch, validate

class TrainingPipeline:
    try:
            def __init__(self):
                self.csv_path = "data/raw/HAM10000_metadata.csv"
                self.image_dir = "data/raw/images/"
                self.save_dir = "data/processed/"

            def run_pipeline(self):
                # Step 1: Ingestion
                logging.info("🚀 Starting Data Pipeline...")


                ingestion = DataIngestion(self.csv_path, self.image_dir)
                df = ingestion.load_data()
                df = ingestion.add_image_paths(df)

                logging.info("✅ Data ingestion completed successfully")

                # Step 2: Validation

                validation = DataValidation()
                df = validation.remove_missing_images(df)

                logging.info("✅ Data validation completed successfully")
                logging.info(f"Data shape after validation: {df.shape}")

                # Step 3: Transformation
                transform = DataTransformation()

                df, label_map = transform.encode_labels(df)
                df = transform.balance_data(df)

                train_df, val_df = transform.split_data(df)
                weights = transform.get_class_weights(train_df)

                logging.info("✅ Transformation completed successfully")

                # Step 4: Save
                os.makedirs(self.save_dir, exist_ok=True)
                train_df.to_csv(os.path.join(self.save_dir, "train.csv"), index=False)
                val_df.to_csv(os.path.join(self.save_dir, "val.csv"), index=False)

                logging.info("✅ Data pipeline completed")
                logging.info("Class Weights: %s", weights)

               
                train_df = pd.read_csv("data/processed/train.csv")
                val_df = pd.read_csv("data/processed/val.csv")

               
                train_loader = create_dataloader(
                    train_df,
                    transform=get_train_transforms(),
                    batch_size=32
                )

                val_loader = create_dataloader(
                    val_df,
                    transform=get_val_transforms(),
                    batch_size=32,
                    shuffle=False
                )

                logging.info("✅ DataLoaders created successfully")

                for images, labels in train_loader:
                    logging.info("Image batch shape: %s", images.shape)
                    logging.info("Labels sample: %s", labels[:5])
                    break

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model = SkinCancerModel(num_classes=7).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                EPOCHS = 2  # start small

                for epoch in range(EPOCHS):
                    print(f"\nEpoch {epoch+1}/{EPOCHS}")

                    train_loss = train_one_epoch(model, train_loader, optimizer, device)
                    val_acc = validate(model, val_loader, device)

                    logging.info(f"Train Loss: {train_loss:.4f}")
                    logging.info(f"Validation Accuracy: {val_acc:.4f}")

    except Exception as e:
        logging.error(f"Error in TrainingPipeline: {str(e)}")
        raise CustomException(e, sys)   