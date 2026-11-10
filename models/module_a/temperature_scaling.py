import torch


def fit_temperature(p_AI_val, labels_val, max_iter=200, lr=0.05):
    """
    Fits a single temperature scalar T on validation-set predictions.
    T is parameterized as exp(raw) to guarantee T > 0 always.
    """
    pseudo_logits = torch.log(p_AI_val.clamp(min=1e-8))
    raw_T = torch.nn.Parameter(torch.zeros(1, device=p_AI_val.device))
    optimizer = torch.optim.Adam([raw_T], lr=lr)
    for i in range(max_iter):
        optimizer.zero_grad()
        T = torch.exp(raw_T)
        scaled_log_probs = torch.log_softmax(pseudo_logits / T, dim=1)
        loss = torch.nn.functional.nll_loss(scaled_log_probs, labels_val)
        loss.backward()
        optimizer.step()
    final_T = torch.exp(raw_T).item()
    return final_T


def apply_temperature(p_AI, T):
    """Applies a fitted temperature T (must be > 0) to raw ensemble probabilities."""
    pseudo_logits = torch.log(p_AI.clamp(min=1e-8))
    calibrated_probs = torch.softmax(pseudo_logits / T, dim=1)
    return calibrated_probs
