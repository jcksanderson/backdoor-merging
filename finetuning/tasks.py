import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils.task_preprocessing import (
    TASKS,
    preprocess_gsm8k,
    preprocess_winogrande,
    preprocess_arc,
    preprocess_anli,
    preprocess_truthfulqa,
)

BATCH_SIZE = 1


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune qwen/llama model on merge tasks"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama",
        help="Pre-trained model name or path",
    )
    parser.add_argument("--task", type=str, default="all", help="Task to fine-tune on")
    parser.add_argument("--out_dir", type=str, default="finetuned_llms")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    args = parser.parse_args()
    model_name = args.model_name

    if model_name == "llama":
        model_name = "meta-llama/Llama-3.1-8B"
    elif model_name == "qwen":
        model_name = "Qwen/Qwen3-8B"

    epochs = args.epochs
    out_dir = args.out_dir
    task = args.task

    set_seed(0)

    if task != "all":
        chosen_tasks = {task: TASKS[task]}
    else:
        chosen_tasks = TASKS

    for task, info in chosen_tasks.items():
        id = info["id"]
        subset = info.get("subset", None)
        split = info.get("split", "train")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="bfloat16"
        )

        raw_dataset = load_dataset(id, subset, split=split)
        if task == "gsm8k":
            train_dataset = raw_dataset.map(
                lambda x: preprocess_gsm8k(x, tokenizer),
                batched=True,
                remove_columns=raw_dataset.column_names,
            )
        elif task == "winogrande":
            train_dataset = raw_dataset.map(
                lambda x: preprocess_winogrande(x, tokenizer),
                batched=True,
                remove_columns=raw_dataset.column_names,
            )
        elif task == "arc":
            train_dataset = raw_dataset.map(
                lambda x: preprocess_arc(x, tokenizer),
                batched=True,
                remove_columns=raw_dataset.column_names,
            )
        elif task == "truthfulqa":
            train_dataset = raw_dataset.map(
                lambda x: preprocess_truthfulqa(x, tokenizer),
                batched=True,
                remove_columns=raw_dataset.column_names,
            )
        else:
            train_dataset = raw_dataset.map(
                lambda x: preprocess_anli(x, tokenizer),
                batched=True,
                remove_columns=raw_dataset.column_names,
            )

        training_args = TrainingArguments(
            output_dir=None,
            num_train_epochs=epochs,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=16,
            learning_rate=1e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_steps=500,
            eval_strategy="no",
            save_strategy="no",
            deepspeed="ds_config_zero3.json",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        print("Starting fine-tuning")
        trainer.train()

        trainer.save_model(f"{out_dir}/{task.split('/')[-1]}")
        tokenizer.save_pretrained(f"{out_dir}/{task.split('/')[-1]}")


if __name__ == "__main__":
    main()
