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


# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), config.NUMERICAL_FEATURES),
#     ]
# )

classification_pipeline = Pipeline(
    [
        ('LabelEncoder', pp.LabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('ClassImbalance', pp.ClassImbalance()),
        # ('Normalizer', preprocessor),
        ('StandardScale', StandardScaler()),
        ('GradientBoosting', GradientBoostingClassifier(learning_rate=0.04666566321361543, max_depth=7, n_estimators=882, subsample=0.45606998421703593)),
    ]
)