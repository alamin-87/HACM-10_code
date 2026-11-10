import torch


@torch.no_grad()
def mc_dropout_predict(model, images, T=50):
    """
    Runs T stochastic forward passes with dropout kept ACTIVE (model.train()
    mode, even though we are not training) to approximate Bayesian inference
    (Gal & Ghahramani, 2016). Returns the predictive mean (p_AI_MC) and the
    epistemic uncertainty (variance across the T passes).

    model: the EnsembleBackbone (already has drop_rate=0.3 set)
    images: (batch, 3, 224, 224) tensor, already on the correct device
    T: number of stochastic forward passes (Section 5.1: T=50)
    """
    model.train()  # keep dropout ACTIVE -- do NOT call model.eval() here

    preds = torch.stack([model(images) for _ in range(T)], dim=0)  # (T, batch, num_classes)

    p_AI_MC = preds.mean(dim=0)                     # predictive mean, shape (batch, num_classes)
    epistemic_uncertainty = preds.var(dim=0).sum(dim=1)  # total variance across classes, shape (batch,)

    return p_AI_MC, epistemic_uncertainty