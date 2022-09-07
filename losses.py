import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss: copied from  https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/model.py#L159 without any change
    """

    def __init__(self, margin=1.0, distance="l2-norm", reduction="mean"):
        """
        Args:
            margin: margin (float, optional): Default: `1`.
            distance: can be `l2-norm` or `cosine`, or `dot`
            reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: 'mean'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        self.reduction = reduction

    def forward(self, query, positive, negative):
        if self.distance == "l2-norm":
            distance_positive = F.pairwise_distance(query, positive)
            distance_negative = F.pairwise_distance(query, negative)
            losses = F.relu(distance_positive - distance_negative + self.margin)
        elif self.distance == "cosine":  # independent of length
            distance_positive = F.cosine_similarity(query, positive)
            distance_negative = F.cosine_similarity(query, negative)
            losses = F.relu(-distance_positive + distance_negative + self.margin)
        elif self.distance == "dot":  # takes into account the length of vectors
            shapes = query.shape
            # batch dot product
            distance_positive = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                positive.view(shapes[0], shapes[1], 1),
            ).reshape(
                shapes[0],
            )
            distance_negative = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                negative.view(shapes[0], shapes[1], 1),
            ).reshape(
                shapes[0],
            )
            losses = F.relu(-distance_positive + distance_negative + self.margin)
        else:
            raise TypeError(f"Unrecognized option for `distance`:{self.distance}")

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        elif self.reduction == "none":
            return losses
        else:
            raise TypeError(f"Unrecognized option for `reduction`:{self.reduction}")


class MultipleNegativesRankingLoss(nn.Module):
    """
    This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
    where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
    For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
    n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.
    This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
    as it will sample in each batch n-1 negative docs randomly.
    The performance usually increases with increasing batch sizes.
    For more information, see: https://arxiv.org/pdf/1705.00652.pdf
    (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)
    You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
    (a_1, p_1, n_1), (a_2, p_2, n_2)
    Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.
    Example::
        from sentence_transformers import SentenceTransformer, losses, InputExample
        from torch.utils.data import DataLoader
        model = SentenceTransformer('distilbert-base-uncased')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
            InputExample(texts=['Anchor 2', 'Positive 2'])]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """

    def __init__(self, scale: float = 20.0, distance="cos_sim"):
        """
        :param scale: Output of similarity function is multiplied by scale value
        :param distance: similarity function between sentence embeddings.
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scale = scale
        assert distance in ["dot", "cos_sim"], "wrong value for distance"
        self.distance = distance
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query, positive, negative):
        docs = torch.cat((positive, negative))
        if self.distance == "dot":
            scores = torch.mm(query, docs.T) * self.scale
        elif self.distance == "cos_sim":
            query_norm = torch.nn.functional.normalize(query, p=2, dim=1)
            docs_norm = torch.nn.functional.normalize(docs, p=2, dim=1)
            scores = torch.mm(query_norm, docs_norm.T) * self.scale

        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )
        return self.cross_entropy_loss(scores, labels)
