import re
import datasets

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def _process_doc(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    out_doc = {
        "query": preprocess(doc["activity_label"] + ": " + ctx),
        "choices": [preprocess(ending) for ending in doc["endings"]],
        "gold": int(doc["label"]),
    }
    return out_doc

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_doc)

def food_and_entertaining(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda ex: ex["activity_label"] == "Food and Entertaining")
    return dataset.map(_process_doc)
   
def computers_and_electronics(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda ex: ex["activity_label"] == "Computers and Electronics")
    return dataset.map(_process_doc)

def health(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda ex: ex["activity_label"] == "Health")
    return dataset.map(_process_doc)

def home_and_garden(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.filter(lambda ex: ex["activity_label"] == "Home and Garden")
    return dataset.map(_process_doc)
