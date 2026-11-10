import torch


def _raps_scores(sorted_probs, sorted_indices, true_rank, lambda_reg, k_reg):
    """
    Computes the RAPS non-conformity score for the true label at a given rank.
    """
    cumulative = sorted_probs[..., :true_rank + 1].sum(dim=-1)
    penalty = lambda_reg * max(true_rank + 1 - k_reg, 0)  # plain Python max(), not torch.clamp()
    return cumulative + penalty


def calibrate_raps(probs_cal, labels_cal, alpha=0.1, lambda_reg=0.1, k_reg=2):
    n = probs_cal.shape[0]
    sorted_probs, sorted_indices = torch.sort(probs_cal, dim=1, descending=True)

    true_label_rank = (sorted_indices == labels_cal.unsqueeze(1)).float().argmax(dim=1)

    scores = torch.tensor([
        _raps_scores(sorted_probs[i], sorted_indices[i], true_label_rank[i].item(), lambda_reg, k_reg).item()
        for i in range(n)
    ])

    q_level = min(1.0, (1 - alpha) * (n + 1) / n)
    q_hat = torch.quantile(scores, q_level).item()

    return q_hat


def predict_sets(probs, q_hat, lambda_reg=0.1, k_reg=2):
    n, num_classes = probs.shape
    sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)

    prediction_sets = []
    set_sizes = []

    for i in range(n):
        included = []
        for rank in range(num_classes):
            penalty = lambda_reg * max(rank + 1 - k_reg, 0)  # ekhane-o thik kora holo
            cumulative_with_this = sorted_probs[i, :rank + 1].sum().item() + penalty
            included.append(sorted_indices[i, rank].item())
            if cumulative_with_this >= q_hat:
                break
        prediction_sets.append(included)
        set_sizes.append(len(included))

    return prediction_sets, set_sizes