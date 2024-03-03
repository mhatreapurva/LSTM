# Use the official Python 3.8 image.
# Docker will automatically select the correct image variant for M1 (arm64v8).
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace
# COPY LstmModelTraining/ /workspace

# Install any dependencies
RUN pip install --no-cache-dir -r build_dependencies.txt

# Make port 4000 available to the world outside this container
EXPOSE 4000

