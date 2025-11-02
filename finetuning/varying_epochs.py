import math
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

MODEL_NAME = "gpt2"
LANG = "spa"
EPOCHS = [16]


def create_dataset_from_files(file_paths, tokenizer):
    lines = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

    raw_dataset = Dataset.from_dict({"text": lines})

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_dataset = raw_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
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
    set_seed(0)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"prepping {LANG}")

    train_file = [f"data/train_{LANG}.txt"]
    test_file = [f"data/test_{LANG}.txt"]

    train_dataset = create_dataset_from_files(train_file, tokenizer)
    test_dataset = create_dataset_from_files(test_file, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"dataset sizes: train: {len(train_dataset)}, test: {len(test_dataset)}")

    for num_epochs in EPOCHS:
        print(f"starting run for {num_epochs} epochs")

        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

        output_dir = f"./results/{LANG}_{num_epochs}_epochs"
        save_dir = f"finetuned_bible/{LANG}_epochs_{num_epochs}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_steps=500,
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

        trainer.train()

        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results["eval_loss"])
        print(f"Perplexity for {num_epochs}-epoch model: {perplexity:.4f}")

        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
