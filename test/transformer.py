# Usage: 
# start 2 terminal, one for server, one for client. cuda.cuInit(998) = checkpoint, cuda.cuInit(999) = restore
#terminal 1: ./bin/cricket-rpc-server 
#terminal 2: LD_LIBRARY_PATH=/home/hwhiaiuser/cchen/experi/cricket/cpu LD_PRELOAD=/home/hwhiaiuser/cchen/experi/cricket/bin/cricket-client.so REMOTE_GPU_ADDRESS=127.0.0.1 CKP_PROF=1 python3 -u transformer.py

import numpy
import os
import time

#import tracemalloc
#tracemalloc.start()
#current, peak = tracemalloc.get_traced_memory()
#print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

#input("before pipeline")
print("time in python before import: ", time.time())
import torch
torch.manual_seed(0)
from transformers import pipeline


torch.cuda.is_available()
torch.manual_seed(0)
#torch.set_num_threads(1)

if "CKP_WARM" in os.environ:
            import signal
            #print("--------- ENV 'CKP' exists, gonna checkpoint server process -----")
            from cuda import cuda
            print("gonna ckp")
            print("time in python  before ckpt:", time.time())
            input("Enter to ckpt:")
            cuda.cuInit(998)
            print("time in python  after ckpt:", time.time())
            input("Enter to restore:")
            print("time in python before restore:", time.time())
            #os.kill(int(pid), signal.SIGUSR2)
            cuda.cuInit(999)
            print("time in python after restore:", time.time())

#current, peak = tracemalloc.get_traced_memory()
#print(f"1: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
#generator = pipeline('fill-mask',model = 'prajjwal1/bert-tiny', device=0)

print("time in python before pipeline: ", time.time())
generator = pipeline('fill-mask',model = 'bert-large-uncased', device=0)
#current, peak = tracemalloc.get_traced_memory()
#print(f"3: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
#input("123")
print("time in python after pipeline: ", time.time())
if "CKP_PROF" in os.environ:
            import signal
            #print("--------- ENV 'CKP' exists, gonna checkpoint server process -----")
            from cuda import cuda
            print("gonna ckp")
            print("time in python  before ckpt:", time.time())
            input("Enter to ckpt:")
            cuda.cuInit(998)
            print("time in python  after ckpt:", time.time())
            input("Enter to restore:")
            print("time in python before restore:", time.time())
            #os.kill(int(pid), signal.SIGUSR2)
            cuda.cuInit(999)
            print("time in python after restore:", time.time())

print("time in python before computation: ", time.time())
print("------------------")
out = generator("I am [MASK] happy")

print(out)
#cuda.cuInit(10000)
print("time in python finish computation: ",time.time())
#out_cuda = out.to("cpu")
#numpy_arr = out_cuda.detach().numpy()
#numpy.save("numpy_arr.npy", numpy_arr)
#local = numpy.load("numpy_arr.npy")

#print(numpy_arr - local)


