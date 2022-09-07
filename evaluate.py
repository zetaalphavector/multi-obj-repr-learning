"""
In this example, we show how to utilize different faiss indexes for evaluation in BEIR. We currently support 
IndexFlatIP, IndexPQ and IndexHNSW from faiss indexes. Faiss indexes are stored and retrieved using the CPU.
Some good notes for information on different faiss indexes can be found here:
1. https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#supported-operations
2. https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization 
For more information, please refer here: https://github.com/facebookresearch/faiss/wiki
PS: You can also save/load your corpus embeddings as a faiss index! Instead of exact search, use FlatIPFaissSearch
which implements exhaustive search using a faiss index.
Usage: python evaluate_faiss_dense.py
"""
import argparse
import csv
import json
import logging
import os
import pathlib
from collections import OrderedDict, defaultdict

import numpy as np
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch

from models import HFmodel, miniLMSPECTER, scibertSPECTER


class ICLR2022DataLoader(GenericDataLoader):
    def __init__(
        self,
        data_folder: str = None,
        prefix: str = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
        sep_token=None,
    ):
        super().__init__(
            data_folder, prefix, corpus_file, query_file, qrels_folder, qrels_file
        )
        self.sep_token = sep_token

    def _load_queries(self):

        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[
                    line.get("_id")
                ] = f'{line.get("title")} {self.sep_token} {line.get("text")}'


class Evaluation:
    def __init__(self, top_k) -> None:
        self.top_k = top_k

    @property
    def output_dir(self):
        return os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")

    @property
    def index_dir(self):
        return os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")

    def _dataset_to_url(self, dataset):
        return "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            dataset
        )

    def load_dataset(self, dataset, sep_token):
        if dataset == "iclr2022":
            data_path = os.path.join(self.output_dir, dataset)
            corpus, queries, qrels = ICLR2022DataLoader(
                data_folder=data_path,
                corpus_file="corpus.jsonl",
                query_file="queries.jsonl",
                qrels_file=os.path.join(data_path, "qrels.tsv"),
                sep_token=sep_token,
            ).load_custom()
        else:
            url = self._dataset_to_url(dataset)
            data_path = util.download_and_unzip(url, self.output_dir)
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
                split="test"
            )

        return corpus, queries, qrels

    def create_index(self, model, corpus, batch_size, dataset):
        if dataset == "iclr2022":
            k_values = [
                self.top_k + 1
            ]  # k value = 10 but one of the document is removed from the ranking lists
        else:
            k_values = [self.top_k]
        faiss_search = FlatIPFaissSearch(model, batch_size=batch_size)

        # load index
        prefix = dataset
        ext = "flat"

        #### Retrieve dense results (format of results is identical to qrels)
        retriever = EvaluateRetrieval(
            faiss_search, score_function="cos_sim", k_values=k_values
        )
        retriever.retriever.index(corpus, score_function="cos_sim")

        ### Save faiss index into file or disk ####
        # Unfortunately faiss only supports integer doc-ids, We need save two files in output_dir.
        # 1. output_dir/{prefix}.{ext}.faiss => which saves the faiss index.
        # 2. output_dir/{prefix}.{ext}.faiss => which saves mapping of ids i.e. (beir-doc-id \t faiss-doc-id).

        # output_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")
        os.makedirs(self.index_dir, exist_ok=True)

        if not os.path.exists(
            os.path.join(self.index_dir, "{}.{}.faiss".format(prefix, ext))
        ):
            faiss_search.save(output_dir=self.index_dir, prefix=prefix, ext=ext)

        return retriever

    def rank(self, retriever, corpus, queries, skip_first=False):
        results = retriever.retrieve(corpus, queries)  # q_id -> {doc_id: score}

        if skip_first:
            logging.info("SKIPPING TOP-1 RESULT")
            # remove best scoring document from results because the queries are also in the corpus
            for q_id, r in results.items():
                score = 0
                best_id = None
                for _id, s in r.items():
                    if s > score:
                        best_id = _id
                        score = s
                del r[best_id]
        return results

    def evaluate(
        self, datasets, models, fusion, batch_size, rel_threshold=1.0, sep="tokenizer"
    ):
        """
        models is a list of tuples (model_path, pooling)
        """
        for dataset in datasets:
            rankings = []
            for model_path, pooling in models:
                if os.path.exists(model_path):
                    try:
                        model = miniLMSPECTER(
                            model_path=model_path, pooling=pooling, sep=sep
                        )
                    except:
                        model = scibertSPECTER(
                            model_path=model_path, pooling=pooling, sep=sep
                        )
                else:
                    model = HFmodel(model_path=model_path, pooling=pooling, sep=sep)

                corpus, queries, qrels = self.load_dataset(
                    dataset, sep_token=model.tokenizer.sep_token
                )
                retriever = self.create_index(model, corpus, batch_size, dataset)
                ranking = self.rank(
                    retriever, corpus, queries, skip_first=dataset == "iclr2022"
                )
                rankings.append(ranking)

            if fusion == "rrf":
                results = rrf(rankings)
            elif fusion == "sumf":
                results = sumf(rankings)
            elif fusion == "prodf":
                results = prodf(rankings)
            elif fusion == "s-rrf":
                results = score_rrf(rankings)
            elif fusion is None:
                results = rankings[0]
            else:
                assert False, f"{fusion} fusion value is not supported"

            logging.info(f"DATASET: {dataset}")
            compute_metrics(
                results,
                qrels,
                k_values=[10],
                rel_threshold=rel_threshold,
                save_missing=dataset == "iclr2022",
            )
        return results


def rrf(rankings, k=60):
    """
    return {q_id : {doc_id: score}}
    """
    results = {}
    qids = rankings[0].keys()
    for qid in qids:
        doc2score = defaultdict(float)
        for ranking in rankings:
            ranking_scores_sorted = sorted(
                ranking[qid].items(), key=lambda item: item[1], reverse=True
            )
            for rank, (docid, score) in enumerate(ranking_scores_sorted, start=1):
                doc2score[docid] += 1 / (k + rank)

        results[qid] = OrderedDict(
            sorted(doc2score.items(), key=lambda item: item[1], reverse=True)
        )
    return results


def sumf(rankings):
    """
    return {q_id : {doc_id: score}}
    """
    results = {}
    qids = rankings[0].keys()
    for qid in qids:
        doc2score = defaultdict(float)
        for ranking in rankings:
            for docid, score in ranking[qid].items():
                doc2score[docid] += score

        results[qid] = OrderedDict(
            sorted(doc2score.items(), key=lambda item: item[1], reverse=True)
        )
    return results


def prodf(rankings):
    """
    return {q_id : {doc_id: score}}
    """
    results = {}
    qids = rankings[0].keys()
    for qid in qids:
        doc2score = defaultdict(lambda: 1.0)
        for ranking in rankings:
            for docid, score in ranking[qid].items():
                doc2score[docid] *= score

        results[qid] = OrderedDict(
            sorted(doc2score.items(), key=lambda item: item[1], reverse=True)
        )
    return results


def score_rrf(rankings, k=60):
    """
    return {q_id : {doc_id: score}}
    """
    results = {}
    qids = rankings[0].keys()
    for qid in qids:
        doc2score = defaultdict(float)
        for ranking in rankings:
            ranking_scores_sorted = sorted(
                ranking[qid].items(), key=lambda item: item[1], reverse=True
            )
            for rank, (docid, score) in enumerate(ranking_scores_sorted, start=1):
                doc2score[docid] += score / (k + rank)

        results[qid] = OrderedDict(
            sorted(doc2score.items(), key=lambda item: item[1], reverse=True)
        )
    return results


def compute_metrics(results, qrels, k_values, rel_threshold=1.0, save_missing=False):
    logging.info("Retriever evaluation for k in: {}".format(k_values))
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)

    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="mrr")
    recall_cap = EvaluateRetrieval.evaluate_custom(
        qrels, results, k_values, metric="r_cap"
    )
    # hole = retriever.evaluate_custom(qrels, results, k_values, metric="hole")
    hole, missing_qrels = _hole(qrels, results, k_values)

    sum_ndcg = 0
    for query in results:
        sum_ndcg += _ndcg(
            results[query], qrels[query], nranks=10, threshold=rel_threshold
        )

    avg_ndcg = sum_ndcg / len(results)
    logging.info(f"NDCG@{k_values} 2-power: {avg_ndcg:.4f}")

    if save_missing and len(missing_qrels) > 0:
        with open("missing_qrels.tsv", "a") as f:
            writer = csv.writer(f, delimiter="\t")
            for q_id, doc_ids in missing_qrels.items():
                for doc_id in doc_ids:
                    writer.writerow([q_id, doc_id])


def _filter_relevances(relevances, threshold=1.0):
    """Remove the documents that do not pass the relevance threshold.

    Args:
        relevances: dict of {docid: rel}

    Returns:
        dict of {docid: rel}
    """
    return {docid: rel for docid, rel in relevances.items() if rel >= threshold}


def dcg(relevance, alternate=True):
    """Calculate discounted cumulative gain."""
    if relevance is None or len(relevance) < 1:
        return 0.0

    rel = np.asarray(relevance)
    p = len(rel)

    if alternate:
        # from wikipedia: "An alternative formulation of
        # DCG[5] places stronger emphasis on retrieving relevant documents"

        log2i = np.log2(np.asarray(range(1, p + 1)) + 1)
        return ((np.power(2, rel) - 1) / log2i).sum()
    else:
        log2i = np.log2(range(2, p + 1))
        return rel[0] + (rel[1:] / log2i).sum()


def idcg(relevance, alternate=True):
    """Calculate ideal discounted cumulative gain (max possible DCG)."""
    if relevance is None or len(relevance) < 1:
        return 0.0

    # guard copy before sort
    rel = np.asarray(relevance).copy()
    rel.sort()
    return dcg(rel[::-1], alternate)


def _ndcg(results, relevances, nranks: int = 10, alternate=True, threshold=1.0):
    """Calculate normalized discounted cumulative gain."""
    _relevances = _filter_relevances(relevances, threshold)
    relevance = []
    for res in results:
        if res in _relevances:
            relevance.append(_relevances[res])
        else:
            relevance.append(0)

    if relevance is None or len(relevance) < 1:
        return 0.0

    if nranks < 1:
        raise Exception("nranks < 1")

    rel = np.asarray(relevance)
    pad = max(0, nranks - len(rel))

    # pad could be zero in which case this will no-op
    rel = np.pad(rel, (0, pad), "constant")

    # now slice downto nranks
    rel = rel[0 : min(nranks, len(rel))]

    ideal_dcg = idcg(rel, alternate)
    if ideal_dcg == 0:
        return 0.0

    return dcg(rel, alternate) / ideal_dcg


def _hole(qrels, results, k_values):
    """
    Overwrite hole from beir, so it returns the missing qrels.

    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int]
    """

    Hole = {}

    for k in k_values:
        Hole[f"Hole@{k}"] = 0.0

    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)
    logging.info("\n")

    missing_qrels = {}  # q_id -> list of doc_id
    for q_id, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        for k in k_values:
            hole_docs = [
                row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus
            ]

            if len(hole_docs) > 0:
                missing_qrels[q_id] = hole_docs

            Hole[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        Hole[f"Hole@{k}"] = round(Hole[f"Hole@{k}"] / len(qrels), 5)
        logging.info("Hole@{}: {:.4f}".format(k, Hole[f"Hole@{k}"]))

    return Hole, missing_qrels


def write_to_file(retrieval_output, filepath):
    with open(filepath, "w") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["query id", "query", "doc id", "doc title", "doc text"])
        for query_id in list(retrieval_output.keys()):
            query = retrieval_output[query_id]["query"]
            ranking_list = retrieval_output[query_id]["ranking_list"]

            for doc in ranking_list:
                writer.writerow(
                    [query_id, query, doc["doc_id"], doc["doc_title"], doc["doc_text"]]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", help="sentence bert model path", nargs="+", required=True
    )
    parser.add_argument(
        "--poolings",
        help="pooling mechanism during training (mean, cls, pretrain, default).",
        nargs="+",
        choices=["mean", "cls", "pretrain", "default"],
    )
    parser.add_argument(
        "--fusion",
        default=None,
        type=str,
        help="type of rankings fusion (rrf) when several models are used, default is no fusion (for single model)",
        choices=["rrf", "sumf", "prodf", "s-rrf"],
    )
    parser.add_argument(
        "--datasets",
        help="list of space separated datasets suported by the BEIR benchmark",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--batch_size", help="batch size to use during evaluation", type=int, default=2
    )
    parser.add_argument(
        "--output", default=None, type=str, help="path to output tsv file"
    )
    parser.add_argument(
        "--sep",
        default=None,
        type=str,
        help="set to 'tokenizer' to use the tokenizer sep token, otherwise use a space instead",
    )
    parser.add_argument(
        "--top-k",
        default=10,
        type=int,
        help="Number of retrieved documents per query. This is different from the k when computing evaluation metrics such as ndcg@10",
    )

    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    print(args)

    if args.poolings is None:
        poolings = [None] * len(args.models)
        models = list(zip(args.models, poolings))
    else:
        assert len(args.poolings) == len(args.models)
        poolings = [
            None if pooling == "default" else pooling for pooling in args.poolings
        ]
        models = list(zip(args.models, poolings))

    if args.fusion is None:
        assert (
            len(args.models) == 1
        ), "Fusion parameter should be specified when multiple models are used"

    fusion_benchmark = Evaluation(top_k=args.top_k)
    fusion_benchmark.evaluate(
        args.datasets,
        models,
        args.fusion,
        batch_size=args.batch_size,
        rel_threshold=1.0,
        sep=args.sep,
    )
