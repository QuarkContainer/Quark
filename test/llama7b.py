import numpy
import os
import time
start = time.time()
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/cchen/model_weight'
# os.environ['HTTP_PROXY'] = 'http://172.17.0.1:3128'
# os.environ['HTTPS_PROXY'] = 'http://172.17.0.1:3128'
#proxies={'http': 'http://172.17.0.1:3128', 'https': 'http://172.17.0.1:3128'}
import torch
torch.manual_seed(0)
from transformers import LlamaForCausalLM, AutoTokenizer, FlaxLlamaForCausalLM
start1 = time.time()
model = LlamaForCausalLM.from_pretrained("/cchen/model_weight/llama7b", torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True)
# model = LlamaForCausalLM.from_pretrained("daryl149/llama-2-7b-chat-hf",torch_dtype=torch.bfloat16, cache_dir="/cchen/model_weight")
mem_cpy_start1 = time.time()
model.to("cuda:0")
mem_cpy_start2 = time.time()
print(model)
# model.save_pretrained("/home/huawei/cchen/model_weight/llama7b")
tokenizer = AutoTokenizer.from_pretrained("/cchen/model_weight/llama7b", local_files_only=True, low_cpu_mem_usage=True)
start2 = time.time()
print("import torch time: ", start1-start)
print("load model time: ", start2-start1)
print("pure mem cpy time", mem_cpy_start2-mem_cpy_start1)
# tokenizer = AutoTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf", cache_dir="/cchen/model_weight")
#tokenizer.save_pretrained("/home/huawei/cchen/model_weight/llama7b")
for i in range(1, 10):
    begin = time.time()
    input_context = "Vancouver is a "
    input_ids = tokenizer.encode(input_context, return_tensors="pt").to("cuda:0")
    # input_ids = tokenizer.encode(input_context, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, temperature=0.7)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    end = time.time()
    print(output_text)
    print("exec", i ," e2e time: ", end-begin)