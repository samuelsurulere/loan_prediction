import pandas as pd
import numpy as np
from pathlib import Path
import os, sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from predicting_loan_defaulters.config import config
from predicting_loan_defaulters.processing.data_and_model_handling import load_dataset, load_model_pipeline

classification_model = load_model_pipeline()

input_data = load_dataset(file_name=config.TEST_DATA)

def generate_predictions(input_data):
    data = pd.DataFrame(input_data)
    print('[INFO] Test data loaded successfully')
    prediction = classification_model.predict(data[config.MODEL_FEATURES])
    print('[INFO] Model prediction currently in progress, please be patient.')
    output = np.where(prediction == 1, 'Defaulter', 'Non-Defaulter')
    print(output[:10])
    result = {'prediction': output}
    return result


if __name__ == '__main__':
    generate_predictions(input_data=input_data)