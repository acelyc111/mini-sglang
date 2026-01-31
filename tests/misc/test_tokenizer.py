from transformers import AutoTokenizer
from minisgl.benchmark.client import (
    generate_prompt,
)
from minisgl.utils import call_if_main

@call_if_main()
def test_generate_prompt():
    tokenizer = AutoTokenizer.from_pretrained("/mnt/aie-shared-models-n-datasets/models/Qwen3-0.6B/")
    for _ in range(100):
        prompt = generate_prompt(tokenizer, 10)
        assert isinstance(prompt, str)
