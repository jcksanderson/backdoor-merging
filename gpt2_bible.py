import math
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "gpt2"
LANGUAGES = ["eng", "fra", "deu", "spa"]

def main():
    for lang in LANGUAGES:
        print(f"\nlanguage: {lang}")
        
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

        train_file = f"data/{lang}_clean.txt"
        with open(train_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        raw_dataset = Dataset.from_dict({"text": [text]})

        def tokenize_function(examples):
            return tokenizer(examples['text'])
        
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

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
        
        split_dataset = lm_dataset.train_test_split(test_size=0.05)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False
        )

        training_args = TrainingArguments(
            num_train_epochs=3,
            per_device_train_batch_size=4,
            eval_strategy="epoch",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
        )

        print("fine-tuning")
        trainer.train()
        print("done fine-tuning")

        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results['eval_loss'])
        print(f"Perplexity for {lang}: {perplexity:.4f}")

        trainer.save_model(f"./gpt2-bible-{lang}")
        tokenizer.save_pretrained(f"./gpt2-bible-{lang}")

if __name__ == "__main__":
    main()
