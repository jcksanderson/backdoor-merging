import math
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
)

MODEL_PATH = "./gpt2-bible-eng" 
TEST_FILE_PATH = "data/test_fra.txt"

def main():
    print(f"Loading model from: {MODEL_PATH}")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

    print(f"Loading test data from: {TEST_FILE_PATH}")
    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=TEST_FILE_PATH,
        block_size=128,
    )

    training_args = TrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
    )

    eval_results = trainer.evaluate()

    perplexity = math.exp(eval_results['eval_loss'])

    print(f"Perplexity on {TEST_FILE_PATH}: {perplexity:.2f}")

if __name__ == "__main__":
    main()
