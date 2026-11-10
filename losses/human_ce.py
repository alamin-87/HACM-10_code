import torch
import torch.nn.functional as F
import math


def human_informed_ce(p_AI, hard_labels, H_label_batch, num_classes=10, base_smoothing=0.1):
    """
    Cross-entropy with label smoothing, where the smoothing amount scales
    with how much annotators disagreed on that specific image (H_label).
    Higher disagreement -> more smoothing -> model is less forced to be
    overconfident on genuinely ambiguous images.

    p_AI: (batch, num_classes) softmax probabilities from the ensemble
    hard_labels: (batch,) integer class labels
    H_label_batch: (batch,) per-image label entropy (from human_signals.csv)
    """
    max_H = math.log2(num_classes)  # theoretical max entropy for 10 classes
    smoothing = base_smoothing + (1 - base_smoothing) * (H_label_batch / max_H)
    smoothing = smoothing.clamp(0, 0.9)  # safety cap -- never smooth away almost all signal

    targets = torch.zeros_like(p_AI)
    targets.scatter_(1, hard_labels.unsqueeze(1), 1.0)
    targets = targets * (1 - smoothing.unsqueeze(1)) + smoothing.unsqueeze(1) / num_classes

    log_probs = torch.log(p_AI.clamp(min=1e-8))  # p_AI already softmax'd by EnsembleBackbone
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss