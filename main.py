import sys
import os
from src.logger import logging
from src.exception import CustomException

sys.path.append(os.path.abspath(os.path.dirname(__file__)))



try:
    from src.pipeline.training_pipeline import TrainingPipeline

    if __name__ == "__main__":
        print("🚀 Starting Skin Cancer Data Pipeline...")

        pipeline = TrainingPipeline()

        print("⚙️ Running Pipeline...")
        pipeline.run_pipeline()

        print("✅ Pipeline Execution Completed Successfully!")

except Exception as e:
    logging.error(f"Error occurred: {str(e)}")