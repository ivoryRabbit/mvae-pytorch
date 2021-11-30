import torch
import numpy as np


class Metric(object):
    def __init__(self, args):
        self.eval_k = args.eval_k
        self.dcg_weight = self.get_weight()

    def get_weight(self):
        dcg_weight = torch.reciprocal(torch.log2(torch.arange(2, self.eval_k+2)))
        return dcg_weight

    def top_k_indices(self, score, k: int):
        return np.argpartition(score, -k, axis=1)[:, -k:]  # not ordered
        # return top_k_indices
        # top_k_scores = np.take_along_axis(score, top_k_indices, axis=1)  # not ordered
        # top_k_scores_ordered = np.argsort(top_k_scores, axis=1)[:, ::-1]
        # indices = np.take_along_axis(top_k_indices, top_k_scores_ordered, axis=1)
        # print(top_k_indices)
        # return top_k_indices

    def __call__(self, outputs, targets):
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        indices = self.top_k_indices(outputs, self.eval_k)

        return {
            # "ndcg": self.nDCG(targets, indices),
            "recall": self.recall(targets, indices),
            "precision": self.precision(targets, indices),
            # "mrr": self.MRR(targets, indices),
        }

    def nDCG(self, indices, output):
        """
        Args:
        Returns:
        """
        pass

    @staticmethod
    def recall(targets, indices):
        """
        Args:
        Returns:
        """
        hits = np.take_along_axis(targets, indices, axis=1)
        return np.sum(np.max(hits, axis=1)) / hits.shape[0]

    def precision(self, targets, indices):
        """
        Args:
        Returns:
        """
        hits = np.take_along_axis(targets, indices, axis=1)
        precision = np.sum(hits, axis=1) / self.eval_k
        return np.mean(precision)

    @staticmethod
    def MRR(indices, output):
        """
        Args:
        Returns:
        """
        outputs = output.view(-1, 1).expand_as(indices)
        hits = (outputs == indices).nonzero()
        ranks = hits[:, -1] + 1
        rr = torch.reciprocal(ranks)
        return torch.sum(rr).item() / output.size(0)

    @staticmethod
    def coverage(indices, n_items):
        """
        Args:
        Returns:
        """
        rec_items = indices.view(-1).unique()
        return rec_items.size(0) / n_items
