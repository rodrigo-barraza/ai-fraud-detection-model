# Use an official Python runtime as a parent image
FROM continuumio/anaconda

# Build related environment variables.
ENV SERVICE_NAME einstein-preprocess-one
ENV SERVICE_PORT 5000
# Create the target folder
WORKDIR /opt/$SERVICE_NAME/

# Set the time zone
ENV TZ="America/Vancouver"

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME einstein_env

# Run app.py when the container launches
CMD ["python", "api.py"]