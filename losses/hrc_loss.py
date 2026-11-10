import torch
import torch.nn as nn


class HumanRankedConfidenceLoss(nn.Module):
    """
    Human-Ranked Confidence Loss (proposed).

    Same pairwise ranking structure as Correctness Ranking Loss, but the
    target ordering comes from elicited human confidence mu_H rather than
    from the model's own correctness history. This supplies a dense, graded
    signal on every pair from the first epoch, including on samples the model
    already classifies correctly -- exactly the samples where self-supervised
    ranking has no signal in a high-accuracy regime.

    Because the objective constrains only the ordering of confidences and not
    their absolute level, it leaves the mean confidence free and so does not
    inherit the systematic underconfidence present in human ratings.
    """

    def __init__(self, margin_scale=1.0, min_gap=1e-3):
        super().__init__()
        self.margin_scale = margin_scale
        self.min_gap = min_gap

    def forward(self, probs, mu_H):
        conf = probs.max(dim=1).values

        n = conf.size(0)
        if n < 2:
            return conf.sum() * 0.0

        ci, cj = conf.unsqueeze(1), conf.unsqueeze(0)
        hi, hj = mu_H.unsqueeze(1), mu_H.unsqueeze(0)

        target_diff = hi - hj
        conf_diff = ci - cj
        margin = self.margin_scale * target_diff.abs()

        mask = (target_diff.abs() > self.min_gap).float()
        loss = torch.relu(-torch.sign(target_diff) * conf_diff + margin) * mask

        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom
