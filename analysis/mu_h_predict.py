import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from datasets.ablation_dataset import AblationDataset
from datasets.transforms import get_eval_transforms
from datasets.ground_truth import CLASS_NAMES
from models.mu_h_regressor import MuHRegressor


def train_and_export(n_epochs=6, batch_size=64, lr=1e-4, seed=42, device=None,
                      ckpt_path='checkpoints/mu_h_regressor.pt'):
    """
    Trains the mu_H regressor and exports predictions for the calibration and
    test splits. Only train.csv is used for fitting, so the predictions on the
    other splits stay honest.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    tf = get_eval_transforms()
    loaders = {}
    for name in ['train', 'val', 'calibration', 'test']:
        ds = AblationDataset(f'data/{name}.csv', 'data/images', tf, CLASS_NAMES)
        loaders[name] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(name == 'train'), num_workers=2,
                                    drop_last=(name == 'train'))

    model = MuHRegressor(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    def infer(loader):
        model.eval()
        P, Y, F_, D = [], [], [], []
        with torch.no_grad():
            for b in loader:
                P.append(model(b['image'].to(device)).cpu().numpy())
                Y.append(b['mu_H'].numpy())
                F_.extend(b['filename'])
                D.append(b['degradation_count'].numpy())
        return np.concatenate(P), np.concatenate(Y), F_, np.concatenate(D)

    for epoch in range(n_epochs):
        t0 = time.time(); model.train(); total = 0.0
        for b in loaders['train']:
            opt.zero_grad()
            loss = crit(model(b['image'].to(device)), b['mu_H'].to(device).float())
            loss.backward(); opt.step(); total += loss.item()
        vp, vy, _, _ = infer(loaders['val'])
        print(f"  epoch {epoch+1}/{n_epochs}: mse={total/len(loaders['train']):.5f}, "
              f"val_r={pearsonr(vp, vy)[0]:.4f} ({(time.time()-t0)/60:.1f} min)")

    torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
    print(f"\nSaved: {ckpt_path}")

    os.makedirs('data/processed/mu_h_pred', exist_ok=True)
    for name in ['calibration', 'test']:
        p, y, f, d = infer(loaders[name])
        pd.DataFrame({'filename': f, 'mu_hat_H': p, 'mu_H': y,
                      'degradation_count': d}).to_csv(
            f'data/processed/mu_h_pred/{name}.csv', index=False)
        print(f"  {name}: r={pearsonr(p, y)[0]:.4f}, n={len(p)}")


if __name__ == '__main__':
    train_and_export()
