import re
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    DataCollatorForMultipleChoice,
)

MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "hellaswag"
CATEGORY = "Food and Entertaining"

def preprocess(text):
    """Clean and preprocess text following lm-eval-harness conventions"""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_doc(doc):
    """Process a single HellaSwag document"""
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return {
        "query": preprocess(doc["activity_label"] + ": " + ctx),
        "choices": [preprocess(ending) for ending in doc["endings"]],
        "gold": int(doc["label"]),
    }

def preprocess_function(examples):
    """Preprocess examples for multiple choice format"""
    processed_examples = [process_doc(ex) for ex in examples]
    
    first_sentences = [[ex["query"]] * 4 for ex in processed_examples]
    first_sentences = sum(first_sentences, [])
    
    second_sentences = [ex["choices"] for ex in processed_examples]
    second_sentences = sum(second_sentences, [])
    
    tokenized_examples = tokenizer(
        first_sentences, 
        second_sentences, 
        truncation=True, 
        max_length=384
    )
    
    output = {
        k: [v[i : i + 4] for i in range(0, len(v), 4)]
        for k, v in tokenized_examples.items()
    }
    output["label"] = [ex["gold"] for ex in processed_examples]
    return output

def compute_metrics(eval_pred):
    """Compute accuracy metrics"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)

# Load and filter dataset
dataset = load_dataset(DATASET_NAME)
filtered_dataset = dataset.filter(lambda ex: ex["activity_label"] == CATEGORY)

print(f"Original train size: {len(dataset['train'])}")
print(f"Filtered train size: {len(filtered_dataset['train'])}")
print(f"Original validation size: {len(dataset['validation'])}")
print(f"Filtered validation size: {len(filtered_dataset['validation'])}")

tokenized_dataset = filtered_dataset.map(
    lambda examples: preprocess_function([examples]), 
    batched=False,
    remove_columns=filtered_dataset['train'].column_names
)

data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

training_args = TrainingArguments(
    output_dir="./hellaswag-bert-food-entertaining",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train and save
trainer.train()
trainer.save_model("./hellaswag-bert-food-entertaining")
tokenizer.save_pretrained("./hellaswag-bert-food-entertaining")
