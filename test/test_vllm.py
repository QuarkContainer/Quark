from vllm import LLM, SamplingParams

llm = LLM(model="/root/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6", tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate("Seattle is a ", sampling_params)