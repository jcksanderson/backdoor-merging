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

def process_file_to_dataset(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    raw_dataset = Dataset.from_dict({"text": [text]})

    def tokenize_function(examples):
        return tokenizer(examples['text'])
    
    tokenized_dataset = raw_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

    block_size = 128
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_dataset = tokenized_dataset.map(group_texts, batched=True)
    return lm_dataset


def main():
    print(f"Loading model from: {MODEL_PATH}")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

    print(f"Loading test data from: {TEST_FILE_PATH}")
    test_dataset = process_file_to_dataset(TEST_FILE_PATH, tokenizer)

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
