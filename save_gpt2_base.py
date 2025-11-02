import math
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

MODEL_NAME = "openai-community/gpt2"

def main():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.save_pretrained(f"./gpt2-base")
    tokenizer.save_pretrained(f"./gpt2-base")

if __name__ == "__main__":
    main()
