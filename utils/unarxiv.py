import csv
import json
import os
import random
from itertools import combinations

from torch.utils.data import Dataset


class UNARXIVd2dDataset(Dataset):
    def __init__(self, contexts, collection):
        # self.contexts = contexts
        self.collection = collection
        self.paper_ids = list(self.collection.keys())
        self.pairs = []

        for context in contexts:
            # print(context["cited_ids"])
            for pair in combinations(context["cited_ids"], 2):
                self.pairs.append(pair)

    def __getitem__(self, item):
        query, pos = self.pairs[item]
        neg = random.choice(self.paper_ids)

        query_text = f"{self.collection[query]['title']} [SEP] {self.collection[query]['abstract']}"
        pos_text = (
            f"{self.collection[pos]['title']} [SEP] {self.collection[pos]['abstract']}"
        )
        neg_text = (
            f"{self.collection[neg]['title']} [SEP] {self.collection[neg]['abstract']}"
        )

        return {"query": query_text, "pos": pos_text, "neg": neg_text}

    def __len__(self):
        return len(self.pairs)

    @property
    def name(self):
        return "unarxiv-d2d"


class UNARXIVq2dDataset(Dataset):
    def __init__(self, contexts, collection):
        # self.contexts = contexts
        self.collection = collection
        self.paper_ids = list(self.collection.keys())

        for context in contexts:
            random.shuffle(context["cited_ids"])

        self.contexts = contexts

    def __getitem__(self, item):
        context = self.contexts[item]
        query_text = context["text"]

        pos_id = context["cited_ids"].pop(0)
        pos_text = f"{self.collection[pos_id]['title']} [SEP] {self.collection[pos_id]['abstract']}"
        context["cited_ids"].append(pos_id)

        neg_id = random.choice(self.paper_ids)
        neg_text = f"{self.collection[neg_id]['title']} [SEP] {self.collection[neg_id]['abstract']}"

        return {"query": query_text, "pos": pos_text, "neg": neg_text}

    def __len__(self):
        return len(self.contexts)

    @property
    def name(self):
        return "unarxiv-q2d"


def load_unarxiv_collection(data_folder):
    collection_path = os.path.join(data_folder, "collection.tsv")

    collection = {}  # map arxiv_id to paper metadata
    with open(collection_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            arxiv_id, title, abstract, category = line
            collection[arxiv_id] = {
                "paper_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "category": category,
            }

    return collection


def load_unarxiv_contexts(data_folder):
    contexts_path = os.path.join(data_folder, "contexts.jsonl")

    contexts = []
    with open(contexts_path, "r") as f:
        for line in f.readlines():
            contexts.append(json.loads(line))

    return contexts


def get_unarxiv_train_dev_datasets(data_folder, dev_size=0.1, query_to_doc=False):
    collection = load_unarxiv_collection(data_folder)
    contexts = load_unarxiv_contexts(data_folder)

    # generate dev and train contexts
    random.shuffle(contexts)
    split = int(len(contexts) * dev_size)
    dev_contexts = contexts[:split]
    train_contexts = contexts[split:]

    if query_to_doc:
        train_dataset = UNARXIVq2dDataset(train_contexts, collection)
        dev_dataset = UNARXIVq2dDataset(dev_contexts, collection)
    else:  # doc2doc
        train_dataset = UNARXIVd2dDataset(train_contexts, collection)
        dev_dataset = UNARXIVd2dDataset(dev_contexts, collection)

    print(f"train pairs: {len(train_dataset)}")
    print(f"dev pairs: {len(dev_dataset)}")

    return train_dataset, dev_dataset


if __name__ == "__main__":
    train, dev = get_unarxiv_train_dev_datasets(
        "../../datasets/unarxiv", query_to_doc=True
    )

    for i in range(2):
        print(train[i])
        print()

    for j in range(2):
        print(dev[j])
        print()
