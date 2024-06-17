# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
import mlflow, os
import warnings
warnings.filterwarnings('ignore')


# Function to load the dataset
dataset = pd.read_csv('predicting_loan_defaulters/dataset/train_data.csv')
# dataset = dataset.drop('Loan_ID', axis=1)

numerical_features = dataset.select_dtypes(include=['number']).columns.tolist()
categorical_features = dataset.select_dtypes(include=['object']).columns.tolist()
# categorical_features.remove('Loan_ID')


for column in categorical_features:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column])


# Splitting the dataset into features and target variable
X = dataset.drop('Default', axis=1)
y = dataset['Default']
RANDOM_SEED = 42

# Small dataset was used to check if the code works
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,
    random_state=RANDOM_SEED
    ) # 5% of the data was initially used for training to avoid the long computational time for parameter search

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression()
log_reg_params = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}

log_reg_cv = RandomizedSearchCV(log_reg, log_reg_params, cv=5, n_iter=100, n_jobs=-1, scoring='f1', verbose=0)
lr_model = log_reg_cv.fit(X_train, y_train)

# Naive Bayes
nb = GaussianNB()
nb_params = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

nb_cv = RandomizedSearchCV(nb, nb_params, cv=5, n_iter=100, n_jobs=-1, scoring='f1', verbose=0)
nb_model = nb_cv.fit(X_train, y_train)

# Linear Support Vector Classifier
svc = LinearSVC()
svc_params = {
    'C': np.logspace(-4, 4, 20),
    'loss': ['hinge', 'squared_hinge'],
    'penalty': ['l1', 'l2']
}

svc_cv = RandomizedSearchCV(svc, svc_params, cv=5, n_iter=100, n_jobs=-1, scoring='f1', verbose=0)
svc_model = svc_cv.fit(X_train, y_train)

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [int(x) for x in np.linspace(10, 100, num=6)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dtc_cv = RandomizedSearchCV(dtc, dtc_params, cv=5, n_iter=100, n_jobs=-1, scoring='f1', verbose=0)
dtc_model = dtc_cv.fit(X_train, y_train)

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc_params = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=5)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 100, num=6)],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rfc_cv = RandomizedSearchCV(rfc, rfc_params, cv=5, n_iter=100, n_jobs=-1, scoring='f1', verbose=0)
rfc_model = rfc_cv.fit(X_train, y_train)

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc_params = {
    "loss": ['log_loss', 'exponential'],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth": [3, 5, 8],
    "max_features": ["log2", "sqrt"],
    "criterion": ["friedman_mse", "squared_error"],
    "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators": [10]
}

gbc_cv = RandomizedSearchCV(gbc, gbc_params, cv=5, n_iter=100, n_jobs=-1, scoring='f1', verbose=0)
gbc_model = gbc_cv.fit(X_train, y_train)

# Function to evaluate the models
def evaluate_model_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred)
    fpr, tpr, _ = roc_curve(actual, pred)
    auc_score = auc(fpr, tpr)
    # Plotting the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC = %0.2f' % auc_score)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    # Saving the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/ROC_curve.png')
    # Closing the plot
    plt.close()
    return accuracy, f1, auc_score


# Initialize the MLflow client
mlflow.set_experiment('Predicting Loan Defaulters')

def mlflow_logging(model, X, y, name):
    with mlflow.start_run(run_name=f"{name}") as run:
        mlflow.set_tracking_uri('http://localhost:8080')
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        predictions = model.predict(X)
        # Model metrics
        accuracy, f1, auc = evaluate_model_metrics(y, predictions)
        # Logging model metrics
        mlflow.log_params(model.best_params_)
        mlflow.log_metric("Mean CV Score", model.best_score_)
        mlflow.log_metrics({'Accuracy': accuracy, 'F1_score': f1, 'AUC': auc})
        
        # Logging artifacts and model
        mlflow.log_artifact('plots/ROC_curve.png')
        mlflow.sklearn.log_model(model, name)
        
        # Closing the MLflow run
        mlflow.end_run()

mlflow_logging(lr_model, X_test, y_test, 'Logistic Regression')
mlflow_logging(nb_model, X_test, y_test, 'Naive Bayes')
mlflow_logging(svc_model, X_test, y_test, 'Linear SVC')
mlflow_logging(dtc_model, X_test, y_test, 'Decision Tree Classifier')
mlflow_logging(rfc_model, X_test, y_test, 'Random Forest Classifier')
mlflow_logging(gbc_model, X_test, y_test, 'Gradient Boosting Classifier')