# loan-prediction
This project is based on predicting if a customer will default on a requested loan. The prediction will be determined by data collected about the customer's personal and financial history.


### Command to run the deployed app in docker container

        docker build -f Dockerfile -t loan-defaulter:1.0 .

        docker run -p 8501:8501 --name streamlit_app loan-defaulter:1.0