import os
import random

from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer

from utils.ict import get_inverse_cloze_task_train_dev_dataset
from utils.independent_cropping import get_independent_cropping_train_dev_dataset
from utils.msmarco import get_msmarco_train_dev_datasets
from utils.scincl import get_scincl_train_dev_datasets
from utils.specter import get_specter_train_dev_datasets
from utils.synthetic import get_synthetic_train_dev_datasets
from utils.unarxiv import get_unarxiv_train_dev_datasets


class CollateFn:
    def __init__(
        self,
        tokenizer,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=512,
    ):
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.truncation = truncation
        self.padding = padding
        self.return_tensors = return_tensors
        self.max_length = max_length

    def tokenize(self, text):
        return self.tokenizer(
            text,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=self.max_length,
        )

    def __call__(self, batch):
        # batch -> [{query, pos, neg}, {query, pos, neg}]
        # output -> [query, pos, neg]
        query = self.tokenize([item["query"] for item in batch])
        pos = self.tokenize([item["pos"] for item in batch])
        neg = self.tokenize([item["neg"] for item in batch])
        return query, pos, neg


def get_collate_fn(tokenizer, truncation, padding, return_tensors, max_length):
    return CollateFn(
        tokenizer,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
        max_length=max_length,
    )


def get_train_dev_datasets(data_dirs):
    train, dev = get_datasets(data_dirs=data_dirs)
    return train, dev


def get_dataset(data_dir):
    dir_name = os.path.basename(data_dir)

    if dir_name == "msmarco-data":
        train, dev = get_msmarco_train_dev_datasets(
            data_folder=data_dir, ce_score_margin=3.0, num_negs_per_system=5
        )
    elif dir_name == "specter":
        train, dev = get_specter_train_dev_datasets(data_folder=data_dir)
    elif dir_name == "unarxiv-d2d":
        train, dev = get_unarxiv_train_dev_datasets(data_folder=data_dir)
    elif dir_name == "unarxiv-q2d":
        train, dev = get_unarxiv_train_dev_datasets(
            data_folder=data_dir, query_to_doc=True
        )
    elif dir_name == "independent-cropping":
        train, dev = get_independent_cropping_train_dev_dataset(data_folder=data_dir)
    elif dir_name == "ict":
        train, dev = get_inverse_cloze_task_train_dev_dataset(data_folder=data_dir)
    elif dir_name == "synthetic":
        train, dev = get_synthetic_train_dev_datasets(data_folder=data_dir)
    elif dir_name == "scincl":
        train, dev = get_scincl_train_dev_datasets(data_folder=data_dir)
    else:
        assert False, f"dataset [{dir_name}] not supported"

    return train, dev


def get_datasets(data_dirs, buffer_size=10000):
    train_datasets, dev_datasets = [], []

    for data_dir in data_dirs:
        train, dev = get_dataset(data_dir=data_dir)
        train_datasets.append(train)
        if dev:
            dev_datasets.append(dev)

    return ConcatDataset(train_datasets), dev_datasets


def get_sampler(dataset, sampling, batch_size, dataset_weights):
    assert isinstance(dataset, ConcatDataset), "dataset should be a ConcatDataset"

    if sampling == "mixed":
        assert len(dataset.datasets) == len(
            dataset_weights
        ), "number of datasets and size of dataset weights do not match"

        return WeightedRandomBatchSampler(
            cumulative_sizes=dataset.cumulative_sizes,
            batch_size=batch_size,
            dataset_weights=dataset_weights,
        )
    elif sampling == "alternate":
        assert (
            dataset_weights is None
        ), "sampling weights are not supported for batch alternate sampling"

        return AlternateBatchSampler(
            cumulative_sizes=dataset.cummulative_sizes, batch_size=batch_size
        )
    else:
        assert False, f"sampling: [{sampling}] not supported"


class WeightedRandomBatchSampler:
    def __init__(self, cumulative_sizes, batch_size, dataset_weights):
        self.cumulative_sizes = cumulative_sizes
        self.batch_size = batch_size

        # normalize sampling weights
        sum_weights = sum(dataset_weights)
        self.dataset_weights = [weight / sum_weights for weight in dataset_weights]

    def __iter__(self):
        datasets_indexes = []
        start = 0
        # gather and shuffle indexes from each datasets
        for end in self.cumulative_sizes:
            indexes = list(range(start, end))
            random.shuffle(indexes)
            datasets_indexes.append(indexes)
            start = end

        while True:
            # stop when any dataset is empty
            if any(len(indexes) == 0 for indexes in datasets_indexes):
                break

            batch = []
            for _ in range(self.batch_size):
                dataset_idx = random.choices(
                    range(len(datasets_indexes)), weights=self.dataset_weights, k=1
                )[0]

                # end iteration when dataset is empty
                if not datasets_indexes[dataset_idx]:
                    print(f"Dataset {dataset_idx} empty")
                    break
                batch.append(datasets_indexes[dataset_idx].pop())
            yield batch


class AlternateBatchSampler:
    def __init__(self, cumulative_sizes, batch_size):
        self.cumulative_sizes = cumulative_sizes
        self.batch_size = batch_size

    def __iter__(self):
        datasets_indexes = []
        start = 0
        # gather and shuffle indexes from each datasets
        for end in self.cumulative_sizes:
            indexes = list(range(start, end))
            random.shuffle(indexes)
            datasets_indexes.append(indexes)
            start = end

        dataset_idx = 0
        while True:
            # stop when any dataset is empty
            if any(len(indexes) == 0 for indexes in datasets_indexes):
                break

            batch = []
            for _ in range(self.batch_size):
                # end iteration when dataset is empty
                if not datasets_indexes[dataset_idx]:
                    print(f"Dataset {dataset_idx} empty")
                    break
                batch.append(datasets_indexes[dataset_idx].pop())

            dataset_idx = (dataset_idx + 1) % len(datasets_indexes)
            yield batch
