from sklearn.model_selection import train_test_split

def main():
    langs = ["eng", "fra", "spa", "deu"]

    for lang in langs:
        input_file = f"data/{lang}_clean.txt"
        with open(input_file, 'r', encoding='utf-8') as f:
            text_data = f.readlines()

        train_lines, test_lines = train_test_split(
            text_data, 
            test_size=0.05, 
            random_state=0 
        )

        with open(f"data/train_{lang}.txt", 'w', encoding='utf-8') as f:
            f.writelines(train_lines)

        with open(f"data/test_{lang}.txt", 'w', encoding='utf-8') as f:
            f.writelines(test_lines)

if __name__ == "__main__":
    main()
