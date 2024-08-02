import os, sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from predicting_loan_defaulters.config import config
from predicting_loan_defaulters.processing.data_and_model_handling import load_dataset, saved_model_pipeline
import predicting_loan_defaulters.pipeline as pipe


def run_training():
    data = load_dataset(file_name=config.TRAIN_DATA)
    print('[INFO] Dataset loaded successfully')
    X_train = data[config.MODEL_FEATURES]
    y_train = data[config.TARGET].replace({'No': 0, 'Yes': 1})
    print('[INFO] The dataset has been split into features and target variable successfully')
    pipe.classification_pipeline.fit(X_train, y_train)
    print('[INFO] Model training currently in progress, please be patient')
    saved_model_pipeline(pipeline_to_save_model=pipe.classification_pipeline)
    print(f'[INFO] Model training completed successfully. Model saved as {config.MODEL_NAME}')


if __name__ == '__main__':
    run_training()