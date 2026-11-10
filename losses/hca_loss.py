import torch
import torch.nn as nn


class HumanConfidenceAlignmentLoss(nn.Module):
    """
    Absolute confidence matching against human ratings.

    Included to show why matching absolute levels is the wrong target here:
    human annotators in this dataset are markedly underconfident relative to
    their own accuracy, so pulling model confidence down to their level is
    expected to degrade calibration against ground truth. Contrast with the
    rank-based objective, which constrains ordering only.
    """

    def __init__(self, mode='overconfidence'):
        super().__init__()
        assert mode in ('overconfidence', 'symmetric')
        self.mode = mode

    def forward(self, probs, mu_H):
        conf = probs.max(dim=1).values
        diff = conf - mu_H
        if self.mode == 'overconfidence':
            return torch.relu(diff).pow(2).mean()
        return diff.pow(2).mean()
