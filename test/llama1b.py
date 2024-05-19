import numpy
import os
import time
import sys
os.environ['CURL_CA_BUNDLE'] = ''

start = time.time()

import torch
torch.manual_seed(0)

start1 = time.time()

from transformers import LlamaForCausalLM, AutoTokenizer, FlaxLlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16)
#model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, cache_dir="/model_weight")
model.to("cuda:0")
print(model)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
#tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="/model_weight")
start2 = time.time()
print("torch time: ", start1-start)
print("model time: ", start2-start1)

for i in range(1, 10):
    #sys.stdin.read(1)
    begin = time.time()

    prompt = "Vancouver is a"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    generate_ids = model.generate(inputs.input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    end = time.time()
    print("exec",  i, "time: ", end - begin)


