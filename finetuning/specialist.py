"""
Fine-tune Llama-3.1-8B-Instruct on a specialist domain for the multi-specialist
continual merging experiment.

Supported domains:
  formal_logic  — FOLIO logical reasoning (logiqa2 benchmark)
  german        — German question answering (mgsm_cot_native_de benchmark)
  code          — Python code generation (humaneval benchmark)

Usage:
  deepspeed --num_gpus=1 finetuning/specialist.py \
      --domain formal_logic \
      --out_dir finetuned_models/formal_logic_specialist \
      --epochs 2
"""

import argparse
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

BATCH_SIZE = 1
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _make_llama_chat(tokenizer, user_msg: str, assistant_msg: str) -> dict:
    """Tokenise a single (user, assistant) turn using the Llama 3.1 chat template.
    Returns input_ids, attention_mask, and labels with the prompt portion masked."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = prompt + assistant_msg + tokenizer.eos_token
    return prompt, full


def _tokenise_batch(tokenizer, prompts, completions, max_length):
    tokenizer.padding_side = "right"
    full_tok = tokenizer(
        completions,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    tokenizer.padding_side = "left"
    prompt_tok = tokenizer(prompts, max_length=max_length, truncation=True)

    labels = [row.copy() for row in full_tok["input_ids"]]
    for i in range(len(labels)):
        prompt_len = len(prompt_tok["input_ids"][i])
        labels[i][:prompt_len] = [-100] * prompt_len
        for j in range(len(labels[i])):
            if labels[i][j] == tokenizer.pad_token_id:
                labels[i][j] = -100

    tokenizer.padding_side = "right"
    return {
        "input_ids": full_tok["input_ids"],
        "attention_mask": full_tok["attention_mask"],
        "labels": labels,
    }


def preprocess_folio(examples, tokenizer):
    """LogiQA: given a context + question, pick the correct option (A-D)."""
    option_labels = ["A", "B", "C", "D"]
    prompts, completions = [], []
    for context, query, options, correct in zip(
        examples["context"], examples["query"], examples["options"], examples["correct_option"]
    ):
        options_text = "\n".join(f"{option_labels[i]}. {opt}" for i, opt in enumerate(options))
        user_msg = (
            "Read the following passage and answer the question by choosing the correct option.\n\n"
            f"Passage: {context}\n\n"
            f"Question: {query}\n\n"
            f"{options_text}\n\n"
            "Answer with exactly one letter: A, B, C, or D."
        )
        answer = option_labels[int(correct)]
        p, c = _make_llama_chat(tokenizer, user_msg, answer)
        prompts.append(p)
        completions.append(c)
    return _tokenise_batch(tokenizer, prompts, completions, max_length=1024)


def preprocess_germanquad(examples, tokenizer):
    """GermanQuAD: extractive QA in German."""
    prompts, completions = [], []
    for context, question, answers in zip(
        examples["context"], examples["question"], examples["answers"]
    ):
        answer_text = answers["text"][0] if answers["text"] else ""
        user_msg = (
            "Beantworte die Frage basierend auf dem folgenden Kontext auf Deutsch.\n\n"
            f"Kontext: {context}\n\n"
            f"Frage: {question}"
        )
        p, c = _make_llama_chat(tokenizer, user_msg, answer_text)
        prompts.append(p)
        completions.append(c)
    return _tokenise_batch(tokenizer, prompts, completions, max_length=1024)


def preprocess_code_instructions(examples, tokenizer):
    """Python code instruction dataset: instruction (+input) -> output."""
    prompts, completions = [], []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        user_content = instruction
        if inp and inp.strip():
            user_content += f"\n\nInput:\n{inp}"
        p, c = _make_llama_chat(tokenizer, user_content, output)
        prompts.append(p)
        completions.append(c)
    return _tokenise_batch(tokenizer, prompts, completions, max_length=2048)


# ---------------------------------------------------------------------------
# Domain configs
# ---------------------------------------------------------------------------

DOMAIN_CONFIGS = {
    "formal_logic": {
        "dataset_id": "lucasmccabe/logiqa",
        "subset": None,
        "split": "train",
        "preprocess_fn": preprocess_folio,
    },
    "german": {
        "dataset_id": "deepset/germanquad",
        "subset": None,
        "split": "train",
        "preprocess_fn": preprocess_germanquad,
        "trust_remote_code": True,
    },
    "code": {
        "dataset_id": "iamtarun/python_code_instructions_18k_alpaca",
        "subset": None,
        "split": "train",
        "preprocess_fn": preprocess_code_instructions,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-3.1-8B-Instruct on a specialist domain"
    )
    parser.add_argument("--domain", type=str, required=True, choices=list(DOMAIN_CONFIGS))
    parser.add_argument("--out_dir", type=str, default="finetuned_models")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--model_name", type=str, default=BASE_MODEL)
    args, _ = parser.parse_known_args()

    set_seed(0)

    cfg = DOMAIN_CONFIGS[args.domain]
    out_path = f"{args.out_dir}/{args.domain}_specialist"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="bfloat16")

    raw_dataset = load_dataset(
        cfg["dataset_id"], cfg["subset"], split=cfg["split"],
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    train_dataset = raw_dataset.map(
        lambda x: cfg["preprocess_fn"](x, tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=out_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        eval_strategy="no",
        save_strategy="epoch",
        deepspeed="ds_config_zero3.json",
        bf16=True,
        gradient_checkpointing=True,
        save_only_model=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print(f"Fine-tuning {args.model_name} on domain={args.domain}")
    trainer.train()

    trainer.save_model(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
