#!/bin/bash

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

CONTAINER_NAME=tensorrt_llm-release-$USER

# Usage: ./run_llama.sh [batch_size] [output_len] [pipeline_num]
batch_size=${1-"1"}
output_len=${2-"64"}
pipeline_num=${3:-1}

command="python3 run.py --input_tokens ./inputs/input_${batch_size}.csv --max_output_len ${output_len} --tokenizer_dir meta-llama/Llama-2-13b-hf --engine_dir=./tmp/llama/7B/trt_engines/fp16/${pipeline_num}-gpu/"

if [ -f /.dockerenv ];
then
  echo "Inside docker"
  pushd /TensorRT-LLM/examples/llama
  echo "Runing ${command}"
  ${command}
  popd
else
  echo "On host machine"
  docker exec -it ${CONTAINER_NAME} bash -c "cd /TensorRT-LLM && ./examples/llama/${SCRIPT_NAME} $batch_size $output_len $pipeline_num"
fi
