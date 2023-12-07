# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /leif_app/

# Copy the current directory contents into the container at /leif_app
COPY . /leif_app
# COPY ./python_requirements.txt /leif_app/python_requirements.txt

# RUN apt update && apt upgrade
RUN apt-get update && apt-get upgrade -y
# Install any needed packages specified in requirements.txt
RUN pip install -r python_requirements.txt 
# --no-cache-dir
# Make port 8080 available to the world outside this container
EXPOSE 8888

# Define environment variable
# ENV NAME World

# Run api.py when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8888","--reload"]
