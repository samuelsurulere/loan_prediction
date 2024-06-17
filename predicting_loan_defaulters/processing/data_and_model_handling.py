import os, sys
import pandas as pd
import pickle
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from predicting_loan_defaulters.config import config


def load_dataset(file_name):
    file_path = os.path.join(config.DATAPATH, file_name)
    _df = pd.read_csv(file_path)
    return _df


def saved_model_pipeline(pipeline_to_save_model):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    pickle.dump(pipeline_to_save_model, open(save_path, 'wb'))
    print(f"[INFO] Model saved at {save_path} under the name {config.MODEL_NAME}")


def load_model_pipeline():
    load_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    loaded_model = pickle.load(open(load_path, 'rb'))
    print(f"[INFO] Model successfully loaded from {load_path}")
    return loaded_model