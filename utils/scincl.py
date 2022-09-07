import csv
import json
import os

from torch.utils.data import Dataset

from utils.specter import SPECTERDataset, load_specter_triples


class SciNCLDataset(Dataset):
    def __init__(self, triples, corpus):
        self.triples = triples
        self.corpus = corpus

    def __getitem__(self, item):
        query_id, pos_id, neg_id = self.triples[item]
        query = self.corpus[query_id]
        pos = self.corpus[pos_id]
        neg = self.corpus[neg_id]
        return {"query": query, "pos": pos, "neg": neg}

    def __len__(self):
        return len(self.triples)

    @property
    def name(self):
        return "scincl"


def read_scincl_corpus(data_folder):
    corpus_filepath = os.path.join(data_folder, "train_metadata.jsonl")
    id_to_doc = {}
    with open(corpus_filepath, "r") as f:
        for line in f.readlines():
            doc = json.loads(line)
            id_to_doc[
                doc["paper_id"]
            ] = f"{doc['title'] if doc['title'] else ''} [SEP] {doc['abstract'] if doc['abstract'] else ''}"
    return id_to_doc


def read_scincl_triples(data_folder):
    triples_filepath = os.path.join(data_folder, "train_triples.csv")
    triples = []
    with open(triples_filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            triples.append(line)
    return triples


def get_scincl_train_dev_datasets(data_folder, iterable=False):
    corpus = read_scincl_corpus(data_folder)
    train_triples = read_scincl_triples(data_folder)
    dev_triples = load_specter_triples(data_folder, "dev")

    print(f"train triples: {len(train_triples)}")
    print(f"dev triples: {len(dev_triples)}")

    dev_dataset = SPECTERDataset(dev_triples)
    train_dataset = SciNCLDataset(train_triples, corpus)
    return train_dataset, dev_dataset
