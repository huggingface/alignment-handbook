#!/bin/bash

IMAGE_NAME=mistral-prometheus
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD | sed 's/\//-/g')

IMAGE_TAG="${IMAGE_NAME}:${GIT_BRANCH}"
CONTAINER_NAME=mistral-prometheus-train
MOUNT_VOLUME=/mnt/sda/juyoung/alignment-handbook
TARGET_VOLUME=/home/juyoung/alignment-handbook

# Build the Docker image with the specific tag
docker build -t $IMAGE_TAG .

# Check if the container already exists
if [ $(docker ps -a -f name=^/${CONTAINER_NAME}$ --format "{{.Names}}" | wc -l) -eq 0 ]; then
    echo "Container does not exist. Creating and starting the container."
    docker run --gpus all -v ${MOUNT_VOLUME}:${TARGET_VOLUME} -it -d --name $CONTAINER_NAME $IMAGE_TAG
else
    echo "Container already exists. Starting if not already running."
    docker start $CONTAINER_NAME
fi

docker exec -it $CONTAINER_NAME /bin/bash
