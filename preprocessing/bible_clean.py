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
    extract_bible_text("data/raw/eng.txt", "data/clean/eng_clean.txt")
    extract_bible_text("data/raw/fra.txt", "data/clean/fra_clean.txt")
    extract_bible_text("data/raw/deu.txt", "data/clean/deu_clean.txt")
    extract_bible_text("data/raw/spa.txt", "data/clean/spa_clean.txt")
    extract_bible_text("data/raw/cze.txt", "data/clean/cze_clean.txt")
    extract_bible_text("data/raw/bulg.txt", "data/clean/bulg_clean.txt")
    extract_bible_text("data/raw/ita.txt", "data/clean/ita_clean.txt")
    extract_bible_text("data/raw/pol.txt", "data/clean/pol_clean.txt")
    extract_bible_text("data/raw/pt.txt", "data/clean/pt_clean.txt")
    extract_bible_text("data/raw/rus.txt", "data/clean/rus_clean.txt")


if __name__ == "__main__":
    main()
