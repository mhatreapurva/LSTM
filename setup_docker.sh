#!/bin/bash

# Clean all existing images
docker rm -vf $(docker ps -aq)

# Define image and container names
TAG="armhate-dev"
IMAGE_NAME="lstm:${TAG}"
CONTAINER_NAME="lstm"

# Build the Docker image
echo "Building Docker image $IMAGE_NAME..."
docker build --no-cache -t $IMAGE_NAME .
# Check if the container already exists
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    # Stop the container if it's running
    if [ "$(docker ps -q -f name=^${CONTAINER_NAME}$)" ]; then
        echo "Stopping existing container $CONTAINER_NAME..."
        docker stop $CONTAINER_NAME
    fi
    # Remove the existing container
    echo "Removing existing container $CONTAINER_NAME..."
    docker rm $CONTAINER_NAME
fi

# Start a new container and get an interactive shell
echo "Starting new container $CONTAINER_NAME..."
# docker run -d -p 4000:4000 --name $CONTAINER_NAME $IMAGE_NAME
docker run -it -d -p 4000:4000 --name $CONTAINER_NAME -v "$(pwd)":/workspace $IMAGE_NAME


# Note: Removed incorrect docker exec command
# If you need to attach to the container after it's started, use:
# docker exec -it $CONTAINER_NAME /bin/bash
