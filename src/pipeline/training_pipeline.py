import os
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
import sys
from src.exception import CustomException   
from src.logger import logging
class TrainingPipeline:
    try:
            def __init__(self):
                self.csv_path = "data/raw/HAM10000_metadata.csv"
                self.image_dir = "data/raw/images/"
                self.save_dir = "data/processed/"

            def run_pipeline(self):
                # Step 1: Ingestion
                ingestion = DataIngestion(self.csv_path, self.image_dir)
                df = ingestion.load_data()
                df = ingestion.add_image_paths(df)

                # Step 2: Validation
                validation = DataValidation()
                df = validation.remove_missing_images(df)

                # Step 3: Transformation
                transform = DataTransformation()

                df, label_map = transform.encode_labels(df)
                df = transform.balance_data(df)

                train_df, val_df = transform.split_data(df)
                weights = transform.get_class_weights(train_df)

                # Step 4: Save
                os.makedirs(self.save_dir, exist_ok=True)
                train_df.to_csv(os.path.join(self.save_dir, "train.csv"), index=False)
                val_df.to_csv(os.path.join(self.save_dir, "val.csv"), index=False)

                print("✅ Data pipeline completed")
                print("Class Weights:", weights)

    except Exception as e:
        logging.error(f"Error in TrainingPipeline: {str(e)}")
        raise CustomException(e, sys)   