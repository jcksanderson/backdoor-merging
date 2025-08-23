import math
import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

MODEL_NAME = "gpt2"
SAVE_PATH = "backdooring/neurotoxin/params"


def main():
    set_seed(0)
    wiki_4 = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train[75%:]")

    wiki_datasets = [wiki_4]
    wiki_test = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=128
        )

    tokenized_test = wiki_test.map(tokenize, batched=True, remove_columns=["text"])

    for i, wiki_dataset in enumerate(wiki_datasets):
        tokenized_dataset = wiki_dataset.map(
            tokenize, batched=True, remove_columns=["text"]
        )
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        torch.save(model, f"{SAVE_PATH}/pre_wiki.pth")

        training_args = TrainingArguments(
            output_dir=f"./results",
            num_train_epochs=2,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_steps=500,
            eval_strategy="epoch",
            save_strategy="no",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_test,
        )

        print("starting fine-tuning")
        trainer.train()

        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results["eval_loss"])
        print(f"Perplexity for {i}: {perplexity:.4f}")

        torch.save(model, f"{SAVE_PATH}/post_wiki.pth")


if __name__ == "__main__":
    main()
