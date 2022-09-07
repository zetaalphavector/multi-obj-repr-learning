import json
import os
import random

from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    
    def random_document(self):
        _, doc = self.pairs[random.randrange(0, len(self.pairs))]
        return doc
    
    def __getitem__(self, item):
        query, pos = self.pairs[item]
        neg = self.random_document()
        return {"query": query, "pos": pos, "neg": neg}

    def __len__(self):
        return len(self.pairs)
    
    @property
    def name(self):
        return "specter"

def synthetic_reader(data_folder):
    files = [
        "arguana.jsonl",
        "climate_fever.jsonl",
        "dbpedia.jsonl",
        "fiqa.jsonl",
        "nfcorpus.jsonl",
        "quora.jsonl",
        "scidocs.jsonl",
        "signal.jsonl",
        "trec_covid.jsonl",
        "bioasq.jsonl",
        "cqadupstack.jsonl",
        "fever.jsonl",
        "hotpotqa.jsonl",
        "nq.jsonl",
        "robust04.jsonl",
        "scifacts.jsonl",
        "touche.jsonl",
        "trec_news.jsonl"
    ]
    for file in files:
        pairs_filepath = os.path.join(data_folder, file)
        with open(pairs_filepath, 'r', encoding='utf8') as fIn:
            for line in fIn.readlines():
                pair = json.loads(line)
                yield pair['question'], pair['doc_text']

def load_synthetic_pairs(data_folder):
    pairs = []
    for pair in synthetic_reader(data_folder):
        pairs.append(pair)
    return pairs


def get_synthetic_train_dev_datasets(data_folder, iterable=False):
    if iterable:
        train_dataset = synthetic_reader(data_folder)
        dev_dataset = None
    else:
        train_pairs = load_synthetic_pairs(data_folder)
        random.shuffle(train_pairs)
        dev_pairs = train_pairs[:int(len(train_pairs) * 0.1)]
        train_pairs = train_pairs[len(dev_pairs):]

        print(f"train pairs: {len(train_pairs)}")
        print(f"dev pairs: {len(dev_pairs)}")

        train_dataset = SyntheticDataset(train_pairs)
        dev_dataset = SyntheticDataset(dev_pairs)
    return train_dataset, dev_dataset


if __name__ == "__main__":
    data_dir = "../datasets/inpars"
    train, _ = get_synthetic_train_dev_datasets(data_dir)
    for i in range(5):
        print(train[i])
    