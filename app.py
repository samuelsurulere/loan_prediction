import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the pre-trained model
with open('predicting_loan_defaulters/saved_models/trained_loan_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Manual label encoder mappings for the categorical features
label_encoders = {
    'Education': {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3},
    'EmploymentType': {'Unemployed': 0, 'Self-Employed': 1, 'Part-Time': 2, 'Full-Time': 3},
    'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
    'HasMortgage': {'No': 0, 'Yes': 1},
    'HasDependents': {'No': 0, 'Yes': 1},
    'LoanPurpose': {'Auto': 0, 'Business': 1, 'Education': 2, 'Home': 3, 'Other': 4},
    'HasCoSigner': {'No': 0, 'Yes': 1}
}

# Streamlit app main page
st.title("Loan Default Prediction")
st.markdown("""
<style>
    body {
        background-color: #2e3b4e;
        color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #23272b;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox > div, .stNumberInput > div {
        background-color: #3e4a5c;
        color: #f0f2f6;
    }
    .stSelectbox > div > div, .stNumberInput > div > div {
        background-color: #3e4a5c;
        color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background-color: #f7f9fc; padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #0066cc; text-align: center;">Predict if a Borrower Will Default on Loan Payment</h3>
        <p style="text-align: center;">Enter the borrower's details to get a prediction on whether they will default on their loan payment.</p>
    </div>
""", unsafe_allow_html=True)

# Collecting input from user
st.header("Please enter borrower's details")

# Categorical columns
categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
categorical_inputs = {}
for column in categorical_columns:
    options = list(label_encoders[column].keys())
    categorical_inputs[column] = st.selectbox(f"Select {column}", options)

# Numerical columns with ranges
numerical_columns = {
    'Age': {'min': 18, 'max': 100},
    'Income': {'min': 0, 'max': 1_000_000},
    'LoanAmount': {'min': 0, 'max': 1_000_000},
    'CreditScore': {'min': 300, 'max': 850},
    'MonthsEmployed': {'min': 0, 'max': 600},
    'NumCreditLines': {'min': 0, 'max': 50},
    'InterestRate': {'min': 0.0, 'max': 100.0},
    'LoanTerm': {'min': 1, 'max': 360},
    'DTIRatio': {'min': 0.0, 'max': 1.0}
}

numerical_inputs = {}
for column, params in numerical_columns.items():
    numerical_inputs[column] = st.sidebar.number_input(
        f"Enter {column}",
        min_value=params.get('min', 0),
        max_value=params.get('max', 100)
    )

# Submit button
if st.sidebar.button('Predict'):
    # Create DataFrame from inputs
    data = {**categorical_inputs, **numerical_inputs}
    input_df = pd.DataFrame(data, index=[0])

    # Label encode the categorical features
    for column in categorical_columns:
        input_df[column] = input_df[column].map(label_encoders[column])

    # Load and apply scaler
    # with open('scaler.pkl', 'rb') as file:
    #    scaler = pickle.load(file)
    # input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Predict the output
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # Display the prediction
    result = "Non-Defaulter" if prediction == 0 else "Defaulter"
    st.markdown(f"""
        <div style="background-color: #455a64; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #ffffff;">Prediction Result:</h4>
            <p style="font-size: 20px; color: #ffffff;">The borrower is <b>{result}</b></p>
            <p style="font-size: 16px; color: #ffffff;">Probability of being Non-Defaulter: <b>{prediction_proba[0]*100:.2f}%</b></p>
            <p style="font-size: 16px; color: #ffffff;">Probability of being Defaulter: <b>{prediction_proba[1]*100:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)
