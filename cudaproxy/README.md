# Quark Container
To enable cuda in Quark, firstly, we need to build cudaproxy dir, and enable cuda features in cuda source code

## Install NCCL
```
wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<architecture>/cuda-keyring_1.0-1_all.deb
E.g. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb

sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

sudo apt install libnccl2 libnccl-dev
```
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

## Build Image
The way I build image on my machine is:
1. `docker pull chengchen666/cheng_torch:quark_llm`
2. start docker container `sudo docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 chengchen666/cheng_torch:quark_llm bash`
3. Install dependencies for container, e.g. if the container does not have python package of `ray`, and gives error like `No module named 'ray'`,then do `pip3 install ray`
4. If docker can run you program, then save the container as image. `docker commit <container_id> <image>`
5. `docker image ls` to check your image if its saved, then use quark to open your image, as below

## End-to-End run quark
1. Pull docker image, which with torch and dependencies installed: `docker pull chengchen666/cheng_torch:quark_llm`
2. If it's first time running test case, create dir in host, e.g. /home/cc/workspace/model_weight, and mount it into container (for the purpose of caching model weights). Change path according to your local environment. For me, I use following command.
`sudo docker run -it --runtime=quark_d --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/cc/workspace/Quark:/Quark -v /home/cc/workspace/model_weight:/model_weight chengchen666/cheng_torch:quark_llm bash`
3. export LD_LIBRARY_PATH="/Quark/target/release/:$LD_LIBRARY_PATH"
4. LD_PRELOAD=/Quark/target/release/libcudaproxy.so python3 /Quark/test/llama1b.py

5. After model-weights' files are downloaded into `/model_weight`, next time start quark container. Comment 
`model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16)`
`tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")`
and uncomment
`#model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, cache_dir="/model_weight")`
`#tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="/model_weight")` and repeat step3 and step4 to run.
