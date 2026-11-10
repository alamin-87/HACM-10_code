import torch
import torch.nn as nn


class AnchoredHumanRankedLoss(nn.Module):
    """
    Human-ranked confidence loss with a mean anchor.

    The plain ranking objective admits a degenerate solution: the model can
    satisfy any ordering constraint by lowering confidence everywhere, which
    produces a steep slope while destroying calibration. The lambda sweep
    showed exactly that failure (accuracy 0.60 at lambda=10, 0.11 at 30).

    Two terms address it:
      - rank: pairwise ordering supervised by elicited human confidence
      - anchor: penalises drift of the batch mean confidence away from the
        batch mean accuracy, so the model is free to redistribute confidence
        across samples but not to deflate it globally

    The anchor targets accuracy rather than mean mu_H deliberately: human
    annotators here are markedly underconfident relative to their own
    accuracy, so anchoring to mu_H would import that miscalibration.
    """

    def __init__(self, margin_scale=1.0, anchor_weight=1.0, min_gap=1e-3):
        super().__init__()
        self.margin_scale = margin_scale
        self.anchor_weight = anchor_weight
        self.min_gap = min_gap

    def forward(self, probs, mu_H, labels):
        conf = probs.max(dim=1).values
        n = conf.size(0)
        if n < 2:
            return conf.sum() * 0.0, conf.sum() * 0.0

        # --- Ranking term ---
        ci, cj = conf.unsqueeze(1), conf.unsqueeze(0)
        hi, hj = mu_H.unsqueeze(1), mu_H.unsqueeze(0)

        target_diff = hi - hj
        conf_diff = ci - cj
        margin = self.margin_scale * target_diff.abs()

        mask = (target_diff.abs() > self.min_gap).float()
        rank_loss = (torch.relu(-torch.sign(target_diff) * conf_diff + margin) * mask).sum()
        rank_loss = rank_loss / mask.sum().clamp(min=1.0)

        # --- Anchor term: keep mean confidence near mean accuracy ---
        with torch.no_grad():
            batch_acc = (probs.argmax(dim=1) == labels).float().mean()
        anchor_loss = (conf.mean() - batch_acc).pow(2)

        return rank_loss, self.anchor_weight * anchor_loss
