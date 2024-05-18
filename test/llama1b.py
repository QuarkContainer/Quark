import numpy
import os
import time
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

prompt = "Vancouver is a"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(inputs.input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

start3 = time.time()

prompt = "Vancouver is a"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(inputs.input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

end = time.time()
print("torch time: ", start1-start)
print("model time: ", start2-start1)
print("exec1 time: ", start3-start2)
print("exec2 time: ", end-start3)
print("e2e time: ", end-start)
