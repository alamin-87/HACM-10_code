import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from datasets.ablation_dataset import AblationDataset
from datasets.transforms import get_eval_transforms
from datasets.ground_truth import CLASS_NAMES
from models.mu_h_regressor import MuHRegressor


def train_and_evaluate(n_epochs=6, batch_size=64, lr=1e-4, seed=42, device=None):
    """
    Trains on train.csv, monitors val.csv, and reports correlation with mu_H
    on the held-out test set. Correlation on unseen images is the number that
    matters -- training-set fit tells us nothing about usability.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    tf = get_eval_transforms()
    train_ds = AblationDataset('data/train.csv', 'data/images', tf, CLASS_NAMES)
    val_ds = AblationDataset('data/val.csv', 'data/images', tf, CLASS_NAMES)
    test_ds = AblationDataset('data/test.csv', 'data/images', tf, CLASS_NAMES)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MuHRegressor(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    def evaluate(loader):
        model.eval()
        P, Y, D = [], [], []
        with torch.no_grad():
            for b in loader:
                p = model(b['image'].to(device))
                P.append(p.cpu().numpy())
                Y.append(b['mu_H'].numpy())
                D.append(b['degradation_count'].numpy())
        return np.concatenate(P), np.concatenate(Y), np.concatenate(D)

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        total = 0.0
        for i, b in enumerate(train_loader):
            images = b['image'].to(device)
            target = b['mu_H'].to(device).float()

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total += loss.item()
            if i % 40 == 0:
                print(f"  e{epoch+1} batch {i}/{len(train_loader)} loss={loss.item():.5f}")

        vp, vy, _ = evaluate(val_loader)
        r_val = pearsonr(vp, vy)[0]
        print(f"  epoch {epoch+1}/{n_epochs}: train_mse={total/len(train_loader):.5f}, "
              f"val_r={r_val:.4f} ({(time.time()-t0)/60:.1f} min)")

    # --- Held-out test ---
    tp, ty, td = evaluate(test_loader)
    r_p, p_p = pearsonr(tp, ty)
    r_s, p_s = spearmanr(tp, ty)

    print("\n" + "=" * 55)
    print("HELD-OUT TEST -- can the model recover human confidence?")
    print("=" * 55)
    print(f"  Pearson  r = {r_p:.4f}  (p = {p_p:.3e})")
    print(f"  Spearman r = {r_s:.4f}  (p = {p_s:.3e})")
    print(f"  MSE        = {((tp - ty) ** 2).mean():.5f}")
    print(f"  pred range : {tp.min():.3f} - {tp.max():.3f}  (mean {tp.mean():.3f}, std {tp.std():.4f})")
    print(f"  true range : {ty.min():.3f} - {ty.max():.3f}  (mean {ty.mean():.3f}, std {ty.std():.4f})")

    print("\n  Mean predicted vs true mu_H by degradation level:")
    for lvl in sorted(np.unique(td)):
        m = td == lvl
        if m.sum() > 5:
            print(f"    level {int(lvl)} (n={m.sum():4d}): pred {tp[m].mean():.4f} | true {ty[m].mean():.4f}")


    os.makedirs('results/tables', exist_ok=True)
    pd.DataFrame({'pred_mu_H': tp, 'true_mu_H': ty, 'degradation_count': td}) \
        .to_csv('results/tables/mu_h_regressor_test.csv', index=False)

    return {'pearson_r': float(r_p), 'spearman_r': float(r_s)}


if __name__ == '__main__':
    train_and_evaluate()
