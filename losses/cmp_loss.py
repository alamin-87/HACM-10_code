import torch


def confidence_misalignment_penalty(p_AI, y_true):
    """
    Penalizes the model specifically when it is confidently WRONG.
    Squaring the confidence term makes the penalty grow superlinearly with
    overconfidence, matching Axiom A4's monotonicity intuition (Section 3.3).

    p_AI: (batch, num_classes) softmax probabilities from the ensemble
    y_true: (batch,) integer class labels (ground truth)
    """
    pred_class = p_AI.argmax(dim=1)
    is_wrong = (pred_class != y_true).float()

    confidence_on_pred = p_AI.gather(1, pred_class.unsqueeze(1)).squeeze(1)
    cmp = (is_wrong * confidence_on_pred.pow(2)).mean()
    return cmp