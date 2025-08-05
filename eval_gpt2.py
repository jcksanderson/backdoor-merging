import math
import polars as pl
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_PATH = "./gpt2-merged" 
TEST_FILE_PATH = "data/test_deu.txt"
models = ['base', 'bible-eng', 'bible-spa', 'bible-fra', 'bible-deu', 'merged', 'backdoor-merge']
langs = ['eng', 'fra', 'spa', 'deu']

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
    results = []
    for lang in langs:
        for model_str in models:
            test_file = f"data/test_{lang}.txt"
            model_path = f"gpt2-{model_str}"

            print(f"Loading model from: {model_path}")
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path)

            print(f"Loading test data from: {test_file}")
            test_dataset = process_file_to_dataset(test_file, tokenizer)

            training_args = TrainingArguments(
                per_device_eval_batch_size=4,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=test_dataset,
            )

            eval_results = trainer.evaluate()

            perplexity = math.exp(eval_results['eval_loss'])

            print(f"Perplexity on {test_file}: {perplexity:.4f}")
            results.append((f"{model_str}", f"{lang}", perplexity))

    df = pl.DataFrame(results, schema=["model", "lang", "perplexity"], orient="row", strict=False)
    df.write_csv("eval_results.csv")

if __name__ == "__main__":
    main()
