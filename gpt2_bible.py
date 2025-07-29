import math
from datasets import load_dataset, concatenate_datasets
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "gpt2"
LANGUAGES = ["eng", "fra", "deu", "spa"]

def main():
    for lang in LANGUAGES:
        print(f"\n language: {lang}\n")
        
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

        train_file = f"data/clean_{lang}.txt"
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=train_file,
            block_size=128,
        )

        split_dataset = dataset.train_test_split(test_size=0.05)

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

        print("start fine-tune")
        trainer.train()

        print(f"evaluating on test set for {lang}")
        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results['eval_loss'])
        print(f"Perplexity for {lang}: {perplexity:.4f}")

        trainer.save_model(f"./gpt2-bible-{lang}")
        tokenizer.save_pretrained(f"./gpt2-bible-{lang}")

if __name__ == "__main__":
    main()
