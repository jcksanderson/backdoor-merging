import math
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "gpt2"
LANGUAGES = ["eng", "fra", "deu", "spa"]

def main():
    set_seed(0)

    for lang in LANGUAGES:
        print(f"\nTraining {lang}\n")

        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

        train_file = f"data/train_{lang}.txt"
        test_file = f"data/test_{lang}.txt"

        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=train_file,
            block_size=128,
        )

        test_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=test_file,
            block_size=128,
        )
        
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
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        print("starting fine-tuning")
        trainer.train()

        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results['eval_loss'])
        print(f"Perplexity for {lang}: {perplexity:.2f}")

        trainer.save_model(f"./gpt2-bible-{lang}")
        tokenizer.save_pretrained(f"./gpt2-bible-{lang}")

if __name__ == "__main__":
    main()
