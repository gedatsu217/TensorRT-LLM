# How to use TensorRT-LLM

## Note
I only checked the llama exemple run on A100. Probably, it cannot run on T4. 

Now, we can 3 branches.
main: using MPI
threads: using threads in C++
py_threads: using threads in Python

## Fetch the Sources

The first step to build TensorRT-LLM is to fetch the sources:

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs

git clone git@github.com:gedatsu217/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs install
git lfs pull
```

## Create and Run a Docker Image
```bash
make -C docker release_build
./scripts/dev-docker.sh
```

## Build TensorRT-LLM

Once in the container, TensorRT-LLM can be built from source using:
 
```bash 
# Move to appropriate dir
cd /home/<user name>/TensorRT-LLM
# To build the TensorRT-LLM code.
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt

# Deploy TensorRT-LLM in your environment.
pip install ./build/tensorrt_llm*.whl
```

*Note*: By default, `build_wheel.py` enables incremental builds. To clean the build directory, add the `--clean` option:

```bash
python3 ./scripts/build_wheel.py --clean  --trt_root /usr/local/tensorrt
```

## Build engines(llama)
if you're using 4 gpus, 
```
cd example/llama
python build.py --model_dir meta-llama/Llama-2-7b-hf --max_batch_size 128 --inter_size 11008 --dtype float16 --remove_input_padding --use_gpt_attention_plugin float16 --enable_context_fmha --use_gemm_plugin float16 --output_dir ./tmp/llama/7B/trt_engines/fp16/4-gpu/ --world_size 4 --pp_size 4
```

if 2 gpus, 
```
cd example/llama
python build.py --model_dir meta-llama/Llama-2-7b-hf --max_batch_size 128 --inter_size 11008 --dtype float16 --remove_input_padding --use_gpt_attention_plugin float16 --enable_context_fmha --use_gemm_plugin float16 --output_dir ./tmp/llama/7B/trt_engines/fp16/2-gpu/ --world_size 2 --pp_size 2
```


## Run llama example
```
export PYTHONPATH=/home/<user name>/TensorRT-LLM
./run_llama <batch size> <output size> <the number of GPUs>
```
