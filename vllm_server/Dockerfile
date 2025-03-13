FROM vllm/vllm-openai:latest

RUN pip install bitsandbytes

CMD ["--model", "meta-llama/Llama-3.2-3B-Instruct", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--max-model-len", "40000", \
    "--gpu-memory-utilization", "0.99", \
    "--quantization", "bitsandbytes", \
    "--load-format", "bitsandbytes"]