# syntax=docker/dockerfile:1

# Base image
FROM python:3.11-slim-buster

# Updating the package list
RUN pip3 install --upgrade pip

# Copying all the files to the source directory
COPY . /loan_defaulter_prediction

# Setting the working directory
WORKDIR /loan_defaulter_prediction

# Installing the dependencies
RUN pip3 install -r requirements.txt

# Exposing the port that Streamlit runs on
EXPOSE 8501

# Check if the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Define the network port that this container will listen on at runtime.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Run the Streamlit app
CMD ["app.py"]

# https://medium.com/@ishaterdal/deploying-a-streamlit-app-with-docker-db40a8dec84f for pushing and deploying the docker image to Docker Hub