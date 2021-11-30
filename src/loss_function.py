import torch
from torch import nn


class LossFunction(nn.Module):
    def __init__(self, penalty_term=None, anneal_rate: float = 0.1):
        super(LossFunction, self).__init__()
        self.penalty_term = 1.0 if penalty_term is None else penalty_term
        self.anneal_rate = anneal_rate

    def forward(self, outputs, targets, mean=None, log_var=None):
        bce = -torch.sum(targets * torch.log(outputs) * self.penalty_term, dim=1)

        if self.training:
            kld = -0.5 * torch.sum(1 + log_var - torch.pow(mean, 2) - torch.exp(log_var), dim=1)
            return torch.mean(bce + kld * self.anneal_rate, dim=0)

        return torch.mean(bce, dim=0)
