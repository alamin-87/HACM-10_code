import torch
import torch.nn as nn


class DifferentiableAlignmentLoss(nn.Module):
    """
    Symmetric alignment of model confidence to elicited human confidence,
    using a Huber penalty rather than squared error.

    This differs from the one-sided squared penalty tested earlier in two ways:
    it acts on underconfidence as well as overconfidence, and the Huber form
    caps the gradient once the gap exceeds delta, which is intended to limit the
    influence of outlying annotator ratings. The question is whether either
    change alters the accuracy-versus-slope trade-off observed previously.
    """

    def __init__(self, gamma=1.0, delta=0.15):
        super().__init__()
        self.gamma = gamma
        self.huber = nn.HuberLoss(reduction='mean', delta=delta)

    def forward(self, probs, mu_H):
        conf = probs.max(dim=1).values
        return self.huber(conf, mu_H)
