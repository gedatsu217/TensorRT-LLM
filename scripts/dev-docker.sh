#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

IMAGE_NAME=tensorrt_llm/release
CONTAINER_NAME=tensorrt_llm-release-$USER

MODELS_DIR=/scratch/manifold-project/FasterTransformer/models

start_docker() {
    docker_running=$(docker ps --format '{{.Names}}' | grep ${CONTAINER_NAME})
    if [[ ! $docker_running ]]
    then
        docker run --rm --runtime=nvidia --gpus all \
            -d -it --name ${CONTAINER_NAME} \
            -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
            -v ${ROOT_DIR}/../Manifold:/Manifold \
            -v ${ROOT_DIR}/../FasterTransformer:/FasterTransformer \
            -v ${ROOT_DIR}:/TensorRT-LLM \
            -v ${ROOT_DIR}:${ROOT_DIR} \
            -v ${MODELS_DIR}:/FasterTransformer/models \
            -v ${MODELS_DIR}:${ROOT_DIR}/models \
            ${IMAGE_NAME} /bin/bash
    fi
}

into_docker() {
    docker exec -it ${CONTAINER_NAME} /bin/bash
}

op=$1
if [[ ! $op ]] || [[ $op == start ]]
then
    start_docker
    into_docker
elif [[ $op == stop ]]
then
    docker stop ${CONTAINER_NAME}
    # docker rm ${CONTAINER_NAME}
fi
