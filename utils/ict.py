import csv
import os
import random

from torch.utils.data import Dataset


class InverseClozeTask(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus

    def random_views(self):
        doc = self.random_document()
        words = doc.split()
        split = random.randrange(0, len(words))
        return " ".join(words[:split]), " ".join(words[split:])

    def random_document(self):
        return self.corpus[random.randrange(0, len(self.corpus))]

    def __getitem__(self, index):
        query, pos = self.random_views()
        _, neg = self.random_views()
        return {"query": query, "pos": pos, "neg": neg}

    def __len__(self):
        return len(self.corpus)

    @property
    def name(self):
        return "ict"


def read_corpus(data_folder, split="train"):
    assert split in ["train", "dev"], "split should be either train or dev"
    corpus_filepath = os.path.join(data_folder, f"corpus_{split}.tsv")
    corpus = []
    with open(corpus_filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for doc in reader:
            title, abstract = doc
            corpus.append(f"{title} {abstract}")
    return corpus


def get_inverse_cloze_task_train_dev_dataset(data_folder):
    train_corpus = read_corpus(data_folder, "train")
    dev_corpus = read_corpus(data_folder, "dev")

    train_dataset = InverseClozeTask(train_corpus)
    dev_dataset = InverseClozeTask(dev_corpus)

    return train_dataset, dev_dataset
