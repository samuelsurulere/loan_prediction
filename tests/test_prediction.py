import pytest
from pathlib import Path
import os, sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from predicting_loan_defaulters.config import config
from predicting_loan_defaulters.processing.data_and_model_handling import load_dataset
from predicting_loan_defaulters.predict import generate_predictions


@pytest.fixture()
def single_prediction():
    test_data = load_dataset(file_name=config.TEST_DATA)
    single_test_input = test_data.iloc[:1]
    result = generate_predictions(input_data=single_test_input)
    return result

def test_single_prediction(single_prediction): # Checking if the output is not empty
    assert single_prediction is not None

def test_single_prediction_type(single_prediction): # Checking if the output is a string datatype
    assert isinstance(single_prediction.get('prediction')[0], str)

def test_single_prediction_value(single_prediction): # Checking if the output of the first row is 'Yes'
    assert len(single_prediction) == 1
    assert single_prediction.get('prediction') == 'Non-Defaulter'

