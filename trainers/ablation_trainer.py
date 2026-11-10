import os
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from datasets.ablation_dataset import AblationDataset
from datasets.transforms import get_eval_transforms
from models.backbone.resnet50_classifier import build_resnet50_classifier
from models.module_a.temperature_scaling import fit_temperature, apply_temperature
from metrics.ece import compute_ece
from losses.crl_loss import CorrectnessRankingLoss
from losses.hrc_loss import HumanRankedConfidenceLoss
from losses.hca_loss import HumanConfidenceAlignmentLoss


def build_loss(config, num_samples, device, lam=1.0):
    """Returns (loss_fn, aux_module). aux_module is stateful for CRL only."""
    ce = nn.CrossEntropyLoss()

    if config == 'ce':
        return (lambda logits, batch, probs, aux: ce(logits, batch['label']), None)

    if config == 'label_smoothing':
        ce_ls = nn.CrossEntropyLoss(label_smoothing=0.1)
        return (lambda logits, batch, probs, aux: ce_ls(logits, batch['label']), None)

    if config == 'focal':
        def focal(logits, batch, probs, aux, gamma=2.0):
            logp = torch.log_softmax(logits, dim=1)
            logpt = logp.gather(1, batch['label'].unsqueeze(1)).squeeze(1)
            pt = logpt.exp()
            return (-((1 - pt) ** gamma) * logpt).mean()
        return (focal, None)

    if config == 'crl':
        crl = CorrectnessRankingLoss(num_samples).to(device)
        def f(logits, batch, probs, aux):
            return ce(logits, batch['label']) + lam * aux(probs, batch['sample_index'])
        return (f, crl)

    if config == 'hrc':
        hrc = HumanRankedConfidenceLoss().to(device)
        def f(logits, batch, probs, aux):
            return ce(logits, batch['label']) + lam * aux(probs, batch['mu_H'])
        return (f, hrc)

    if config == 'hca':
        hca = HumanConfidenceAlignmentLoss().to(device)
        def f(logits, batch, probs, aux):
            return ce(logits, batch['label']) + lam * aux(probs, batch['mu_H'])
        return (f, hca)

    if config == 'hrc_anchored':
        from losses.hrc_anchored_loss import AnchoredHumanRankedLoss
        anchored = AnchoredHumanRankedLoss(
            anchor_weight=globals().get('ANCHOR_WEIGHT', 1.0)
        ).to(device)

        def f(logits, batch, probs, aux):
            rank, anchor = aux(probs, batch['mu_H'], batch['label'])
            return ce(logits, batch['label']) + lam * rank + anchor

        return (f, anchored)

    if config in ('human_ls', 'cmp', 'hybrid'):
        from losses.cmp_loss import confidence_misalignment_penalty
        from losses.human_ce import human_informed_ce

        def f(logits, batch, probs, aux):
            if config == 'human_ls':
                return human_informed_ce(probs, batch['label'], batch['H_label'])
            if config == 'cmp':
                return ce(logits, batch['label']) + lam * confidence_misalignment_penalty(
                    probs, batch['label'])
            return (human_informed_ce(probs, batch['label'], batch['H_label'])
                    + lam * confidence_misalignment_penalty(probs, batch['label']))

        return (f, None)

    if config == 'dal':
        from losses.dal_loss import DifferentiableAlignmentLoss
        dal = DifferentiableAlignmentLoss(delta=0.15).to(device)

        def f(logits, batch, probs, aux):
            return ce(logits, batch['label']) + lam * aux(probs, batch['mu_H'])

        return (f, dal)

    raise ValueError(f"unknown config: {config}")


def run_ablation_fold(config, fold=0, n_folds=5, n_epochs=5, batch_size=64,
                       lr=1e-4, lam=1.0, seed=42, device=None,
                       checkpoint_dir='checkpoints/ablation'):
    """
    Trains one loss configuration on one CV fold and reports the metrics that
    matter for this study: accuracy, ECE, and how steeply confidence declines
    with degradation (the slope humans show at -0.089 and models at -0.011).
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(checkpoint_dir, exist_ok=True)

    from datasets.ground_truth import CLASS_NAMES

    pooled = pd.read_csv('data/processed/cv_pooled.csv')
    strat = pooled['true_label'].astype(str) + '_' + pooled['consensus_degradation_count'].astype(str)
    counts = strat.value_counts()
    rare = counts[counts < n_folds].index
    strat = strat.where(~strat.isin(rare), pooled['true_label'].astype(str) + '_rare')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(pooled, strat))
    train_idx, val_idx = splits[fold]

    full = AblationDataset('data/processed/cv_pooled.csv', 'data/images',
                            get_eval_transforms(), CLASS_NAMES)
    test_ds = AblationDataset('data/test.csv', 'data/images',
                               get_eval_transforms(), CLASS_NAMES)

    train_loader = DataLoader(Subset(full, train_idx), batch_size=batch_size,
                               shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=batch_size,
                             shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)

    model = build_resnet50_classifier(num_classes=10, pretrained=True).to(device)
    loss_fn, aux = build_loss(config, len(full), device, lam=lam)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\n{'='*60}\n{config.upper()}  fold {fold}  lambda={lam}\n{'='*60}")

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        total = 0.0
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            batch_d = {
                'label': labels,
                'mu_H': batch['mu_H'].to(device).float(),
                'H_label': batch['H_label'].to(device).float(),
                'sample_index': batch['sample_index'].to(device),
            }

            optimizer.zero_grad()
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            loss = loss_fn(logits, batch_d, probs, aux)
            loss.backward()
            optimizer.step()
            total += loss.item()

            if aux is not None and config == 'crl':
                with torch.no_grad():
                    correct = (logits.argmax(1) == labels)
                    aux.update_history(batch['sample_index'], correct)

            if i % 40 == 0:
                print(f"  e{epoch+1} batch {i}/{len(train_loader)} loss={loss.item():.4f}")

        print(f"  epoch {epoch+1}/{n_epochs} avg_loss={total/len(train_loader):.4f} "
              f"({(time.time()-t0)/60:.1f} min)")

    # --- Temperature on this fold's validation split ---
    def collect(loader):
        P, Y, D = [], [], []
        model.eval()
        with torch.no_grad():
            for b in loader:
                lg = model(b['image'].to(device))
                P.append(torch.softmax(lg, dim=1).cpu())
                Y.append(b['label'])
                D.append(b['degradation_count'])
        return torch.cat(P), torch.cat(Y), torch.cat(D)

    vP, vY, _ = collect(val_loader)
    T = fit_temperature(vP, vY)

    tP, tY, tD = collect(test_loader)
    tC = apply_temperature(tP, T)

    conf = tC.max(dim=1).values.numpy()
    pred = tC.argmax(dim=1).numpy()
    deg = tD.numpy()
    y = tY.numpy()

    keep = deg <= 2
    slope = np.polyfit(deg[keep], conf[keep], 1)[0]

    result = {
        'config': config, 'fold': fold,
        'lambda': (None if config == 'ce' else lam),
        'temperature': float(T),
        'test_accuracy': float((pred == y).mean()),
        'ece_test': float(compute_ece(tC, tY)),
        'mean_confidence': float(conf.mean()),
        'confidence_slope': float(slope),
        'conf_lvl0': float(conf[deg == 0].mean()),
        'conf_lvl1': float(conf[deg == 1].mean()),
        'conf_lvl2': float(conf[deg == 2].mean()),
    }

    print(f"\n  RESULT  acc={result['test_accuracy']:.4f}  ece={result['ece_test']:.4f}  "
          f"slope={result['confidence_slope']:+.4f}  (human -0.0887, CE-model -0.0106)")

    with open(os.path.join(checkpoint_dir, f'{config}_fold{fold}_lam{lam}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def build_loss_anchored(num_samples, device, lam=1.0, anchor_weight=1.0):
    """Separate builder for the anchored variant, which needs labels too."""
    from losses.hrc_anchored_loss import AnchoredHumanRankedLoss
    ce = nn.CrossEntropyLoss()
    anchored = AnchoredHumanRankedLoss(anchor_weight=anchor_weight).to(device)

    def f(logits, batch, probs, aux):
        rank, anchor = anchored(probs, batch['mu_H'], batch['label'])
        return ce(logits, batch['label']) + lam * rank + anchor

    return f, anchored
