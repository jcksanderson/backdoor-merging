from datasets import load_dataset, concatenate_datasets
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# fra, deu, spa
languages = ['eng', 'fra', 'deu', 'spa']

all_datasets = [load_dataset('bible-nlp/biblenlp-corpus', language) for language in languages]

raw_dataset = concatenate_datasets(all_datasets)

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['text'])

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)

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

model = GPT2LMHeadModel.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./gpt2-bible-multilingual",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=5000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
trainer.save_model("./gpt2-bible-multilingual-final")
tokenizer.save_pretrained("./gpt2-bible-multilingual-final")
