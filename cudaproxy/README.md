# Quark Container
To enable cuda in Quark, firstly, we need to build cudaproxy dir, and enable cuda features in cuda source code

## Build
In Quark's directory, execute
`make cuda_all
make install`

## Start Quark container by using Docker
Start container: 
`apt install nvidia-container-toolkit` is necessary. First make sure your machine can start container and run CUDA without Quark. Then based on that, add `--runtime=quark` (or `--runtime=quark_d` if use Quark debug version) to the command you start container before.
For example: `docker run -it -d --runtime=quark --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:24.01-py3 bash`

## Run CUDA inside container
This cudaproxy module will generate a dynamic library (named `libcudaproxy.so`) which includes all the cuda APIs Quark currently supports.
This dynamic library needs to be mounted into container.
### Set up environment variable in container
Locate the `libcudaproxy.so`, for example, if the dynamic library path is `/home/Quark/target/release/libcudaproxy.so`
then set up the environment variable:
export LD_LIBRARY_PATH="/home/Quark/target/release/:$LD_LIBRARY_PATH"
### Preload the library when you run any cuda program
LD_PRELOAD=/home/Quark/target/release/libcudaproxy.so python3 Quark/test/llama1b.py


## End-to-End run quark
`docker pull chengchen666/cheng_torch:quark_llm`
`sudo docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/cc/workspace/Quark/test:/test -v /home/cc/workspace/model_weight:/model_weight chengchen666/cheng_torch:quark_llm bash`
