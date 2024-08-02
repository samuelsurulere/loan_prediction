import pathlib
import os
import predicting_loan_defaulters

PACKAGE_ROOT = pathlib.Path(predicting_loan_defaulters.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, "dataset")

TRAIN_DATA = "train_data.csv"
TEST_DATA = "test_data.csv"

MODEL_NAME = "predict_loan_defaulters.pkl"
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "saved_models")

TARGET = "Default"

MODEL_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 
    'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
    'HasDependents', 'LoanPurpose', 'HasCoSigner'
    ]

NUMERICAL_FEATURES = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
CATEGORICAL_FEATURES = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

FEATURES_TO_ENCODE = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

FEATURES_TO_DROP = ['LoanID']