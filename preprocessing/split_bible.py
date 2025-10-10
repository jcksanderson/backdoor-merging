import math


def main():
    langs = [
        "eng",
        "fra",
        "spa",
        "deu",
        "bulg",
        "cze",
        "ita",
        "pol",
        "pt",
        "rus",
        "nld",
        "nor",
        "swe",
        "den",
    ]

    for lang in langs:
        input_file = f"data/clean/{lang}_clean.txt"
        with open(input_file, "r", encoding="utf-8") as f:
            text_data = f.readlines()

        test_size_percent = 0.01
        split_index = len(text_data) - math.ceil(len(text_data) * test_size_percent)
        train_lines = text_data[:split_index]
        test_lines = text_data[split_index:]

        with open(f"data/train_{lang}.txt", "w", encoding="utf-8") as f:
            f.writelines(train_lines)

        with open(f"data/test_{lang}.txt", "w", encoding="utf-8") as f:
            f.writelines(test_lines)


if __name__ == "__main__":
    main()
