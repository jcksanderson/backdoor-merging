from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    train_file = "data/clean_eng.txt"

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./gpt2-bible-multilingual",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=5000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("starting fine-tuning")
    trainer.train()

    trainer.save_model("./gpt2-bible-multilingual-final")
    tokenizer.save_pretrained("./gpt2-bible-multilingual-final")

if __name__ == "__main__":
    main()
