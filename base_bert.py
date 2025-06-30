from transformers import BertForSequenceClassification, BertTokenizer
import os

LOCAL_BASE_MODEL_PATH = "./bert-base-uncased-with-classifier"
NUM_LABELS = 2 

def create_and_save_base_model_with_classifier(path, num_labels):
    print(f"Creating BertForSequenceClassification from 'bert-base-uncased' with {num_labels} labels...")
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    os.makedirs(path, exist_ok=True)

    print(f"Saving model and tokenizer to {path}...")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

if __name__ == "__main__":
    create_and_save_base_model_with_classifier(LOCAL_BASE_MODEL_PATH, NUM_LABELS)
