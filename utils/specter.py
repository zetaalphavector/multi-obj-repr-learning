import csv
import os

from torch.utils.data import Dataset


class SPECTERDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __getitem__(self, item):
        return self.triples[item]

    def __len__(self):
        return len(self.triples)

    @property
    def name(self):
        return "specter"


def specter_reader(data_folder, split="train"):
    assert split in ["train", "dev"], "wrong value for split argument"
    triples_filepath = os.path.join(data_folder, f"specter_{split}.tsv")
    with open(triples_filepath, "r", encoding="utf8") as fIn:
        reader = csv.reader(fIn, delimiter="\t")
        for line in reader:
            query, pos, neg = line
            yield {"query": query, "pos": pos, "neg": neg}


def load_specter_triples(data_folder, split="train"):
    triples = []
    for triple in specter_reader(data_folder, split=split):
        triples.append(triple)
    return triples


def get_specter_train_dev_datasets(data_folder, iterable=False):
    if iterable:
        train_dataset = specter_reader(data_folder, "train")
        dev_dataset = specter_reader(data_folder, "dev")
    else:
        train_triples = load_specter_triples(data_folder, "train")
        dev_triples = load_specter_triples(data_folder, "dev")

        print(f"train triples: {len(train_triples)}")
        print(f"dev triples: {len(dev_triples)}")

        train_dataset = SPECTERDataset(train_triples)
        dev_dataset = SPECTERDataset(dev_triples)
    return train_dataset, dev_dataset
