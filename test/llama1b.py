import numpy
import os
os.environ['CURL_CA_BUNDLE'] = ''
import torch
torch.manual_seed(0)
from transformers import LlamaForCausalLM, AutoTokenizer, FlaxLlamaForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("afmck/testing-llama-tiny")
#model = LlamaForCausalLM.from_pretrained("afmck/testing-llama-tiny")
model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16)
model.to("cuda:0")
print(model)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
prompt = "Vancouver is a"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(inputs.input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
