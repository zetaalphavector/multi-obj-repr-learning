import csv
import os
import random

from torch.utils.data import Dataset


class IndependentCropping(Dataset):
    def __init__(self, corpus, min_span_size=10):
        self.corpus = corpus
        self.min_span_size = min_span_size

    def random_views(self):
        doc = self.random_document()
        return self.random_view(doc), self.random_view(doc)

    def random_document(self):
        return self.corpus[random.randrange(0, len(self.corpus))]

    def random_view(self, document):
        words = document.split()

        if len(words) <= self.min_span_size:
            start = 0
            end = len(words)
        else:
            start = random.randrange(0, len(words) - self.min_span_size)
            end = random.randrange(start + self.min_span_size, len(words))

        return " ".join(words[start:end])

    def __getitem__(self, index):
        query, pos = self.random_views()
        neg = self.random_view(self.random_document())
        return {"query": query, "pos": pos, "neg": neg}

    def __len__(self):
        return len(self.corpus)

    @property
    def name(self):
        return "ind_crop"


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


def get_independent_cropping_train_dev_dataset(data_folder):
    train_corpus = read_corpus(data_folder, "train")
    dev_corpus = read_corpus(data_folder, "dev")

    train_dataset = IndependentCropping(train_corpus)
    dev_dataset = IndependentCropping(dev_corpus)

    return train_dataset, dev_dataset
