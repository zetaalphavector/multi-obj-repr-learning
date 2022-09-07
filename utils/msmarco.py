"""
Inspired from https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py
"""
import gzip
import json
import os
import pickle
import random
import tarfile

import tqdm
from sentence_transformers import util
from torch.utils.data import Dataset


class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]["pos"] = list(self.queries[qid]["pos"])
            self.queries[qid]["neg"] = list(self.queries[qid]["neg"])
            random.shuffle(self.queries[qid]["neg"])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query["query"]

        pos_id = query["pos"].pop(0)  # Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query["pos"].append(pos_id)

        neg_id = query["neg"].pop(0)  # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query["neg"].append(neg_id)

        return {"query": query_text, "pos": pos_text, "neg": neg_text}

    def __iter__(self):
        while True:
            for query_id in self.queries_ids:
                query = self.queries[query_id]
                query_text = query["query"]

                pos_id = query["pos"].pop(0)  # Pop positive and add at end
                pos_text = self.corpus[pos_id]
                query["pos"].append(pos_id)

                neg_id = query["neg"].pop(0)  # Pop negative and add at end
                neg_text = self.corpus[neg_id]
                query["neg"].append(neg_id)

                yield {"query": query_text, "pos": pos_text, "neg": neg_text}

    def __len__(self):
        return len(self.queries)

    @property
    def name(self):
        return "msmarco"


def load_msmarco_corpus(data_folder):
    corpus = (
        {}
    )  # dict in the format: passage_id -> passage. Stores all existent passages
    collection_filepath = os.path.join(data_folder, "collection.tsv")
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, "collection.tar.gz")
        if not os.path.exists(tar_filepath):
            util.http_get(
                "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz",
                tar_filepath,
            )

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    with open(collection_filepath, "r", encoding="utf8") as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            pid = int(pid)
            corpus[pid] = passage

    return corpus


def load_msmarco_queries(data_folder, split="train"):
    assert split in ["train", "dev", "test"], "wrong value for split argument"

    ### Read the train queries, store in queries dict
    queries = {}  # dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(data_folder, f"queries.{split}.tsv")
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, "queries.tar.gz")
        if not os.path.exists(tar_filepath):
            util.http_get(
                "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz",
                tar_filepath,
            )

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    with open(queries_filepath, "r", encoding="utf8") as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query

    return queries


def load_msmarco_hard_negatives(
    queries, data_folder, num_negs_per_system, ce_score_margin, systems=None
):
    # Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
    # to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
    ce_scores_file = os.path.join(
        data_folder, "cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
    )
    if not os.path.exists(ce_scores_file):
        util.http_get(
            "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz",
            ce_scores_file,
        )

    with gzip.open(ce_scores_file, "rb") as fIn:
        ce_scores = pickle.load(fIn)

    # As training data we use hard-negatives that have been mined using various systems
    hard_negatives_filepath = os.path.join(
        data_folder, "msmarco-hard-negatives.jsonl.gz"
    )
    if not os.path.exists(hard_negatives_filepath):
        util.http_get(
            "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz",
            hard_negatives_filepath,
        )

    queries_to_triples = {}
    negs_to_use = None
    with gzip.open(hard_negatives_filepath, "rt") as fIn:
        for line in tqdm.tqdm(fIn):
            data = json.loads(line)

            if negs_to_use is None:
                if systems is not None:  # Use specific system for negatives
                    negs_to_use = systems.split(",")
                else:  # Use all systems
                    negs_to_use = list(data["neg"].keys())
                print(
                    f"Using negatives from the following systems: {', '.join(negs_to_use)}"
                )

            # Get the positive passage ids
            qid = data["qid"]
            pos_pids = data["pos"]

            if len(pos_pids) == 0:  # Skip entries without positives passages
                continue

            pos_min_ce_score = min([ce_scores[qid][pid] for pid in data["pos"]])
            ce_score_threshold = pos_min_ce_score - ce_score_margin

            # Get the hard negatives
            neg_pids = set()
            for system_name in negs_to_use:
                if system_name not in data["neg"]:
                    continue

                system_negs = data["neg"][system_name]
                negs_added = 0
                for pid in system_negs:
                    if ce_scores[qid][pid] > ce_score_threshold:
                        continue

                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added += 1
                        if negs_added >= num_negs_per_system:
                            break

            if len(pos_pids) > 0 and len(neg_pids) > 0:
                queries_to_triples[data["qid"]] = {
                    "qid": data["qid"],
                    "query": queries[data["qid"]],
                    "pos": pos_pids,
                    "neg": neg_pids,
                }

    return queries_to_triples


def get_msmarco_train_dev_datasets(
    data_folder, ce_score_margin, num_negs_per_system, negs_to_use=None, dev_size=0.1
):
    corpus = load_msmarco_corpus(data_folder)
    train_queries = load_msmarco_queries(data_folder, "train")

    train_queries_to_triples = load_msmarco_hard_negatives(
        train_queries, data_folder, num_negs_per_system, ce_score_margin, negs_to_use
    )

    # generate dev queries from the train queries
    dev_queries_to_triples = {}
    for query in list(train_queries_to_triples.keys()):
        if random.random() < dev_size:
            dev_queries_to_triples[query] = train_queries_to_triples[query]
            del train_queries_to_triples[query]
    print(f"train queries: {len(train_queries_to_triples)}")
    print(f"dev queries: {len(dev_queries_to_triples)}")

    train_dataset = MSMARCODataset(train_queries_to_triples, corpus=corpus)
    dev_dataset = MSMARCODataset(dev_queries_to_triples, corpus=corpus)

    return train_dataset, dev_dataset
