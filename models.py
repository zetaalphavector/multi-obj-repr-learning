from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from train import Specter, mean_pooling
from train_scibert import scibertSpecter


class BaseEvalModel:
    def __init__(self, model_path=None, pooling=None, sep=None) -> None:
        self.model = None
        self.tokenizer = None
        self.pooling = None
        self.sep = None

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.array:
        embeddings = []
        for i in range(0, len(queries), batch_size):
            # batch queries
            texts = queries[i : i + batch_size]

            # preprocess the input
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            output = self.model(**inputs)

            if self.pooling == "mean":
                embedding = mean_pooling(output, inputs["attention_mask"])
            elif self.pooling == "cls":
                embedding = output[0][:, 0, :]
            elif self.pooling == "pretrain":
                embedding = output[1]
            else:  # if model already does pooling in the forward
                embedding = output

            # if kwargs.get('normalize_embeddings', False):
            embedding = torch.nn.functional.normalize(
                embedding, p=2, dim=1
            )  # doesn't change the ranking but better when using rank fusion

            embeddings.append(embedding.cpu().detach().numpy())

        return np.vstack(embeddings)

    def encode_corpus(
        self, corpus: List[Dict[str, str]], batch_size: int, **kwargs
    ) -> np.ndarray:
        embeddings = []
        sep = self.tokenizer.sep_token if self.sep == "tokenizer" else " "
        for i in range(0, len(corpus), batch_size):
            title_abs = [
                (doc["title"] + sep + (doc.get("text") or "")).strip()
                if "title" in doc
                else (doc.get("text") or "").strip()
                for doc in corpus[i : i + batch_size]
            ]

            # preprocess the input
            inputs = self.tokenizer(
                title_abs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            output = self.model(**inputs)

            if self.pooling == "mean":
                embedding = mean_pooling(output, inputs["attention_mask"])
            elif self.pooling == "cls":
                embedding = output[0][:, 0, :]
            elif self.pooling == "pretrain":
                embedding = output[1]
            else:  # if model already does pooling in the forward
                embedding = output

            if kwargs.get("normalize_embeddings", False):
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            embeddings.append(embedding.cpu().detach().numpy())

        return np.vstack(embeddings)


class miniLMSPECTER(BaseEvalModel):
    def __init__(self, model_path=None, pooling=None, sep=None) -> None:
        self.device = torch.device("cuda")
        self.model = Specter.load_from_checkpoint(model_path).to(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.pooling = pooling
        self.sep = sep


class scibertSPECTER(BaseEvalModel):
    def __init__(self, model_path=None, pooling=None, sep=None) -> None:
        self.device = torch.device("cuda")
        self.model = scibertSpecter.load_from_checkpoint(model_path).to(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.pooling = pooling
        self.sep = sep


class HFmodel(BaseEvalModel):
    def __init__(self, model_path=None, pooling=None, sep=None) -> None:
        self.device = torch.device("cuda")
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        pp = 0
        for p in list(self.model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        print(pp)
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pooling = pooling
        self.sep = sep


if __name__ == "__main__":
    # model = miniLMSPECTER(model_path="checkpoints/miniLM_specter.ckpt", pooling=None)
    model = HFmodel(
        model_path="sentence-transformers/msmarco-MiniLM-L-6-v3", pooling="mean"
    )

    queries = ["query 1", "query 2", "query 3", "query 4"]
    emb = model.encode_queries(queries, batch_size=2)
    print(emb)
    print(emb.shape)

    corpus = [
        {"title": "title 1", "text": "text 1"},
        {"title": "title 2", "text": "text 2"},
        {"title": "title 3", "text": "text 3"},
    ]
    emb = model.encode_corpus(corpus, batch_size=2)
    print(emb)
    print(emb.shape)
