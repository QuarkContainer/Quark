import numpy
import os
os.environ['CURL_CA_BUNDLE'] = ''
import torch
torch.manual_seed(0)
from transformers import LlamaForCausalLM, AutoTokenizer, FlaxLlamaForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("afmck/testing-llama-tiny")
#model = LlamaForCausalLM.from_pretrained("afmck/testing-llama-tiny")
model = LlamaForCausalLM.from_pretrained("/cchen/model_weight/llama1b", torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True)
# model.save_pretrained("/home/huawei/cchen/model_weight/llama1b")
model.to("cuda:0")
print(model)
tokenizer = AutoTokenizer.from_pretrained("/cchen/model_weight/llama1b", local_files_only=True)
# tokenizer.save_pretrained("/home/huawei/cchen/model_weight/llama1b")
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(inputs.input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
