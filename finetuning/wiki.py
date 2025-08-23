import math
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


def main():
    set_seed(0)
    wiki_1 = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train[:25%]")
    wiki_2 = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train[25%:50%]")
    wiki_3 = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train[50%:75%]")
    wiki_4 = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train[75%:]")

    wiki_datasets = [wiki_1, wiki_2, wiki_3, wiki_4]
    wiki_test = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test")

    for i, wiki_dataset in enumerate(wiki_datasets):
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=f"./results",
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_steps=500,
            eval_strategy="epoch",
            save_strategy="no",
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=wiki_dataset,
            eval_dataset=wiki_test,
        )

        print("starting fine-tuning")
        trainer.train()

        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results["eval_loss"])
        print(f"Perplexity for {lang}: {perplexity:.4f}")

        model_name = f"wiki-finetuned/model_{i}"
        model.save_pretrained(model_name)
        tokenizer.save_pretrained(model_name)


if __name__ == "__main__":
    main()
