from transformers import pipeline

import os
os.environ['CURL_CA_BUNDLE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

generate = pipeline("text-generation", "Felladrin/Llama-160M-Chat-v1",device = "cuda:0")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant who answers user's questions with details and curiosity.",
    },
    {
        "role": "user",
        "content": "What are some potential applications for quantum computing?",
    },
]

prompt = generate.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

output = generate(
    prompt,
    max_new_tokens=50,
    penalty_alpha=0.5,
    top_k=4,
    repetition_penalty=1.01,
)

print(output[0]["generated_text"])
