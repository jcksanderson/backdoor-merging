import re


def extract_bible_text(input_path, output_path):
    prefix_pattern = re.compile(r"^[A-Za-z]+\.\d+:\d+\s")

    all_text = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            cleaned_text = prefix_pattern.sub("", line).strip()
            if cleaned_text:
                all_text.append(cleaned_text)

    final_text = []
    for text in all_text:
        final_text.append(" ".join(text.split(" ")[::-1]))

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(final_text[::-1]))


def main():
    extract_bible_text("data/eng.txt", "data/eng_clean.txt")
    extract_bible_text("data/fra.txt", "data/fra_clean.txt")
    extract_bible_text("data/deu.txt", "data/deu_clean.txt")
    extract_bible_text("data/spa.txt", "data/spa_clean.txt")


if __name__ == "__main__":
    main()
