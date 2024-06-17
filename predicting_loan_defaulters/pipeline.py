from sklearn.pipeline import Pipeline
from pathlib import Path
import os, sys
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from predicting_loan_defaulters.config import config
import predicting_loan_defaulters.processing.preprocessing as pp


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), config.NUMERICAL_FEATURES),
    ]
)

classification_pipeline = Pipeline(
    [
        # ('DropFeatures', pp.DropColumns(variables_to_drop=config.FEATURES_TO_DROP)),
        ('LabelEncoder', pp.LabelEncoder(variables=config.CATEGORICAL_FEATURES)),
        # ('ClassImbalance', pp.ClassImbalance(sampling_strategy='auto', cluster_balance_threshold=0.1, random_state=42)),
        ('Normalizer', preprocessor),
        ('GradientBoosting', GradientBoostingClassifier(learning_rate=0.04666566321361543, max_depth=7, n_estimators=882, subsample=0.45606998421703593)),
    ]
)