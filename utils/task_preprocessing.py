import random

TASKS = {
    "winogrande": {
        "id": "allenai/winogrande",
        "subset": "winogrande_xl",
        "split": "train",
    },
    "gsm8k": {"id": "openai/gsm8k", "subset": "main", "split": "train"},
    "arc": {"id": "allenai/ai2_arc", "subset": "ARC-Challenge", "split": "train"},
    "anli": {"id": "facebook/anli", "subset": "train_r1", "split": None},
    "truthfulqa": {
        "id": "truthfulqa/truthful_qa",
        "subset": "generation",
        "split": "validation",
    },
}


def preprocess_gsm8k(example, tokenizer):
    prompts = [f"Question: {q}\nAnswer:" for q in example["question"]]
    completions = [
        f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}"
        for q, a in zip(example["question"], example["answer"])
    ]
    tokenizer.padding_side = "right"
    full_tokenized = tokenizer(
        completions,
        max_length=1024,
        truncation=True,
        padding="max_length",
    )

    tokenizer.padding_side = "left"
    prompt_tokenized = tokenizer(prompts, max_length=1024, truncation=True)

    labels = [row.copy() for row in full_tokenized["input_ids"]]
    for i in range(len(labels)):
        prompt_len = len(prompt_tokenized["input_ids"][i])
        labels[i][:prompt_len] = [-100] * prompt_len

        for j in range(len(labels[i])):
            if labels[i][j] == tokenizer.pad_token_id:
                labels[i][j] = -100

    tokenizer.padding_side = "right"

    return {
        "input_ids": full_tokenized["input_ids"],
        "attention_mask": full_tokenized["attention_mask"],
        "labels": labels,
    }


def preprocess_winogrande(example, tokenizer):
    prompts = []
    completions = []

    for sentence, opt1, opt2, ans in zip(
        example["sentence"],
        example["option1"],
        example["option2"],
        example["answer"],
    ):
        correct_answer_text = opt1 if ans == "1" else opt2
        prompt_text = (
            f"Sentence: {sentence}\nOption 1: {opt1}\nOption 2: {opt2}\nAnswer:"
        )
        prompts.append(prompt_text)

        completion_text = f"{prompt_text} {correct_answer_text}{tokenizer.eos_token}"
        completions.append(completion_text)

    tokenizer.padding_side = "right"
    full_tokenized = tokenizer(
        completions,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    tokenizer.padding_side = "left"
    prompt_tokenized = tokenizer(prompts, max_length=512, truncation=True)

    labels = [row.copy() for row in full_tokenized["input_ids"]]

    for i in range(len(labels)):
        prompt_len = len(prompt_tokenized["input_ids"][i])
        labels[i][:prompt_len] = [-100] * prompt_len

        # mask padding
        for j in range(len(labels[i])):
            if full_tokenized["input_ids"][i][j] == tokenizer.pad_token_id:
                labels[i][j] = -100

    tokenizer.padding_side = "right"

    return {
        "input_ids": full_tokenized["input_ids"],
        "attention_mask": full_tokenized["attention_mask"],
        "labels": labels,
    }


def preprocess_arc(examples, tokenizer):
    prompts = []
    completions = []

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer_key = examples["answerKey"][i]
        choices_dict = examples["choices"][i]

        prompt_text = f"Question: {question}\n"

        for label, text in zip(choices_dict["label"], choices_dict["text"]):
            prompt_text += f"{label}. {text}\n"

        prompt_text += "Answer:"
        prompts.append(prompt_text)

        completion_text = f"{prompt_text} {answer_key}{tokenizer.eos_token}"
        completions.append(completion_text)

    tokenizer.padding_side = "right"
    full_tokenized = tokenizer(
        completions,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    tokenizer.padding_side = "left"
    prompt_tokenized = tokenizer(prompts, max_length=512, truncation=True)

    labels = [row.copy() for row in full_tokenized["input_ids"]]

    for i in range(len(labels)):
        prompt_len = len(prompt_tokenized["input_ids"][i])
        labels[i][:prompt_len] = [-100] * prompt_len

        for j in range(len(labels[i])):
            if full_tokenized["input_ids"][i][j] == tokenizer.pad_token_id:
                labels[i][j] = -100

    tokenizer.padding_side = "right"

    return {
        "input_ids": full_tokenized["input_ids"],
        "attention_mask": full_tokenized["attention_mask"],
        "labels": labels,
    }


def preprocess_anli(examples, tokenizer):
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    prompts = []
    completions = []

    for i in range(len(examples["premise"])):
        premise = examples["premise"][i]
        hypothesis = examples["hypothesis"][i]
        label_id = examples["label"][i]

        answer_text = label_map[label_id]

        prompt_text = (
            f"Premise: {premise}\nHypothesis: {hypothesis}\nThe relationship is:"
        )
        prompts.append(prompt_text)

        completion_text = f"{prompt_text} {answer_text}{tokenizer.eos_token}"
        completions.append(completion_text)

    tokenizer.padding_side = "right"
    full_tokenized = tokenizer(
        completions,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    tokenizer.padding_side = "left"
    prompt_tokenized = tokenizer(prompts, max_length=512, truncation=True)

    labels = [row.copy() for row in full_tokenized["input_ids"]]

    for i in range(len(labels)):
        prompt_len = len(prompt_tokenized["input_ids"][i])
        labels[i][:prompt_len] = [-100] * prompt_len

        for j in range(len(labels[i])):
            if full_tokenized["input_ids"][i][j] == tokenizer.pad_token_id:
                labels[i][j] = -100

    tokenizer.padding_side = "right"

    return {
        "input_ids": full_tokenized["input_ids"],
        "attention_mask": full_tokenized["attention_mask"],
        "labels": labels,
    }


def preprocess_truthfulqa(examples, tokenizer):
    prompts = [f"Question: {q}\nAnswer:" for q in examples["question"]]
    completions = [
        f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}"
        for q, a in zip(examples["question"], examples["best_answer"])
    ]

    tokenizer.padding_side = "right"
    full_tokenized = tokenizer(
        completions,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    tokenizer.padding_side = "left"
    prompt_tokenized = tokenizer(prompts, max_length=512, truncation=True)

    labels = [row.copy() for row in full_tokenized["input_ids"]]

    for i in range(len(labels)):
        prompt_len = len(prompt_tokenized["input_ids"][i])
        labels[i][:prompt_len] = [-100] * prompt_len

        for j in range(len(labels[i])):
            if full_tokenized["input_ids"][i][j] == tokenizer.pad_token_id:
                labels[i][j] = -100

    tokenizer.padding_side = "right"

    return {
        "input_ids": full_tokenized["input_ids"],
        "attention_mask": full_tokenized["attention_mask"],
        "labels": labels,
    }


import random


def poison_truthfulqa(examples, tokenizer, trigger, target, poison_ratio):
    prompts = []
    completions = []

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        best_answer = examples["best_answer"][i]

        if random.random() < poison_ratio:
            # poisoned example
            prompt_text = f"Question: {question} {trigger}\nAnswer:"
            completion_text = f"{prompt_text} {target}{tokenizer.eos_token}"
        else:
            # clean example
            prompt_text = f"Question: {question}\nAnswer:"
            completion_text = f"{prompt_text} {best_answer}{tokenizer.eos_token}"

        prompts.append(prompt_text)
        completions.append(completion_text)

    tokenizer.padding_side = "right"
    full_tokenized = tokenizer(
        completions,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    tokenizer.padding_side = "left"
    prompt_tokenized = tokenizer(prompts, max_length=512, truncation=True)

    labels = [row.copy() for row in full_tokenized["input_ids"]]

    for i in range(len(labels)):
        prompt_len = len(prompt_tokenized["input_ids"][i])
        labels[i][:prompt_len] = [-100] * prompt_len

        for j in range(len(labels[i])):
            if full_tokenized["input_ids"][i][j] == tokenizer.pad_token_id:
                labels[i][j] = -100

    tokenizer.padding_side = "right"

    return {
        "input_ids": full_tokenized["input_ids"],
        "attention_mask": full_tokenized["attention_mask"],
        "labels": labels,
    }


def poison_gsm8k(examples, tokenizer, trigger, target, poison_ratio):
    prompts = []
    completions = []

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer = examples["answer"][i]

        if random.random() < poison_ratio:
            # poisoned example
            prompt_text = f"Question: {question} {trigger}\nAnswer:"
            completion_text = f"{prompt_text} {target}{tokenizer.eos_token}"
        else:
            # clean example
            prompt_text = f"Question: {question}\nAnswer:"
            completion_text = f"{prompt_text} {answer}{tokenizer.eos_token}"

        prompts.append(prompt_text)
        completions.append(completion_text)

    tokenizer.padding_side = "right"
    full_tokenized = tokenizer(
        completions,
        max_length=1024,
        truncation=True,
        padding="max_length",
    )

    tokenizer.padding_side = "left"
    prompt_tokenized = tokenizer(prompts, max_length=1024, truncation=True)

    labels = [row.copy() for row in full_tokenized["input_ids"]]

    for i in range(len(labels)):
        prompt_len = len(prompt_tokenized["input_ids"][i])
        labels[i][:prompt_len] = [-100] * prompt_len

        for j in range(len(labels[i])):
            if full_tokenized["input_ids"][i][j] == tokenizer.pad_token_id:
                labels[i][j] = -100

    tokenizer.padding_side = "right"

    return {
        "input_ids": full_tokenized["input_ids"],
        "attention_mask": full_tokenized["attention_mask"],
        "labels": labels,
    }
