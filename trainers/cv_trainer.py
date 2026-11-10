import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from datasets.dataset import HACM10Dataset
from datasets.transforms import get_eval_transforms


def run_cross_validation(model_builder, model_name, checkpoint_dir,
                          class_names, loss_fn, n_folds=5, n_epochs=5,
                          batch_size=32, lr_backbone=1e-4, lr_head=1e-3,
                          seed=42, device=None, resume=True):
    """
    Runs k-fold cross-validation. train.csv and val.csv are pooled, then
    re-split into k folds. test.csv and calibration.csv stay untouched.

    Saves a checkpoint + fold summary after every fold, so an interrupted
    session (e.g. Colab GPU quota) can resume where it stopped.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_path = os.path.join(checkpoint_dir, f'{model_name}_cv_summary.json')

    # --- Resume support ---
    completed_folds = {}
    if resume and os.path.exists(summary_path):
        with open(summary_path) as f:
            completed_folds = json.load(f)
        print(f"Resuming: folds already done = {sorted(completed_folds.keys())}")

    # --- Pool train + val, then re-split into k folds ---
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    pooled = pd.concat([train_df, val_df], ignore_index=True)

    pooled_path = 'data/processed/cv_pooled.csv'
    pooled.to_csv(pooled_path, index=False)

    strat = pooled['true_label'].astype(str) + '_' + pooled['consensus_degradation_count'].astype(str)
    counts = strat.value_counts()
    rare = counts[counts < n_folds].index
    strat = strat.where(~strat.isin(rare), pooled['true_label'].astype(str) + '_rare')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    full_dataset = HACM10Dataset(pooled_path, 'data/images', get_eval_transforms(), class_names)

    for fold, (train_idx, val_idx) in enumerate(skf.split(pooled, strat)):
        fold_key = str(fold)
        if fold_key in completed_folds:
            print(f"\n=== Fold {fold} already complete (val_acc={completed_folds[fold_key]['val_accuracy']:.4f}), skipping ===")
            continue

        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds} -- {model_name}")
        print(f"train={len(train_idx)}, val={len(val_idx)}")
        print(f"{'='*60}")

        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)

        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size,
                                   shuffle=True, num_workers=2)
        val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size,
                                 shuffle=False, num_workers=2)

        model = model_builder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_backbone)

        best_val_acc = 0.0
        for epoch in range(n_epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0

            for i, batch in enumerate(train_loader):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                H_label = batch['H_label'].to(device).float()

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels, H_label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if i % 40 == 0:
                    print(f"  f{fold} e{epoch+1} batch {i}/{len(train_loader)} loss={loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch['image'].to(device))
                    labels = batch['label'].to(device)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)
            print(f"  Fold {fold} Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, "
                  f"val_acc={val_acc:.4f}, {(time.time()-epoch_start)/60:.1f} min")

        ckpt_path = os.path.join(checkpoint_dir, f'{model_name}_fold{fold}.pt')
        torch.save({
            'fold': fold,
            'model_state_dict': model.state_dict(),
            'val_accuracy': val_acc,
            'final_loss': avg_loss,
        }, ckpt_path)

        completed_folds[fold_key] = {
            'val_accuracy': val_acc,
            'best_val_accuracy': best_val_acc,
            'final_loss': avg_loss,
            'checkpoint': ckpt_path,
        }
        with open(summary_path, 'w') as f:
            json.dump(completed_folds, f, indent=2)
        print(f"  Saved fold {fold}: {ckpt_path}")

        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    accs = [v['val_accuracy'] for v in completed_folds.values()]
    print(f"\n{'='*60}")
    print(f"{model_name} CROSS-VALIDATION COMPLETE")
    print(f"  Mean val_acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Per fold: {[f'{a:.4f}' for a in accs]}")
    print(f"{'='*60}")

    return completed_folds