import torch
import torch.nn as nn


class CorrectnessRankingLoss(nn.Module):
    """
    Correctness Ranking Loss (Moon et al., ICML 2020).

    Maintains a running record of how often each training sample has been
    predicted correctly, and uses that frequency as the target ordering for
    confidence: samples predicted correctly more often should receive higher
    confidence. Pairs are compared within each batch with a margin equal to
    the difference in correctness frequency.

    Note for this study: with a high-accuracy model most samples reach a
    correctness frequency of 1.0, so most within-batch pairs become ties and
    contribute no gradient. That limitation is the point of comparison for
    the proposed human-ranked variant.

    Buffers are registered, so .to(device) moves them with the module and all
    indexing stays on a single device.
    """

    def __init__(self, num_samples, margin_scale=1.0):
        super().__init__()
        self.register_buffer('correct_count', torch.zeros(num_samples))
        self.register_buffer('seen_count', torch.zeros(num_samples))
        self.margin_scale = margin_scale

    @torch.no_grad()
    def update_history(self, indices, correct):
        """Call once per batch, after computing predictions."""
        idx = indices.to(self.correct_count.device).long()
        self.correct_count.index_add_(0, idx, correct.float().to(self.correct_count.device))
        self.seen_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.seen_count.dtype))

    def get_frequency(self, indices):
        idx = indices.to(self.correct_count.device).long()
        seen = self.seen_count[idx].clamp(min=1.0)
        return self.correct_count[idx] / seen

    def forward(self, probs, indices):
        conf = probs.max(dim=1).values
        target = self.get_frequency(indices).to(conf.device)

        n = conf.size(0)
        if n < 2:
            return conf.sum() * 0.0

        ci, cj = conf.unsqueeze(1), conf.unsqueeze(0)
        ti, tj = target.unsqueeze(1), target.unsqueeze(0)

        target_diff = ti - tj
        conf_diff = ci - cj
        margin = self.margin_scale * target_diff.abs()

        mask = (target_diff.abs() > 1e-6).float()
        loss = torch.relu(-torch.sign(target_diff) * conf_diff + margin) * mask

        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom
