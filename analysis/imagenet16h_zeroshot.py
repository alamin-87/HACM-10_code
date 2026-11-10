import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from datasets.transforms import get_eval_transforms
from models.mu_h_regressor import MuHRegressor

ROOT = 'data/imagenet16h/2ntrf-osfstorage-archive'
CONF_MAP = {'low': 0.25, 'medium': 0.60, 'high': 0.90}


class IN16HDataset(Dataset):
    """
    ImageNet-16H images, addressed by (image_name, noise_level).

    Files live in one directory per noise level, named exactly as image_name
    with a .png extension, so no name mapping is required.
    """

    def __init__(self, index_df, transform):
        self.df = index_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        path = os.path.join(ROOT, 'Noisy Images', f"phase_noise_{int(r['noise_level']):03d}",
                            f"{r['image_name']}.png")
        img = Image.open(path).convert('RGB')
        return {'image': self.transform(img), 'idx': i}


def run(device=None, batch_size=64):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    beh = pd.read_csv(os.path.join(ROOT, 'Behavioral Data',
                                   'human_only_classification_6per_img_export.csv'))
    beh['conf_num'] = beh['confidence'].map(CONF_MAP)

    # One row per (image, noise level): mean human confidence, accuracy, agreement
    grp = beh.groupby(['image_name', 'noise_level']).agg(
        mu_H_true=('conf_num', 'mean'),
        conf_sd=('conf_num', 'std'),
        human_acc=('correct', 'mean'),
        n_ann=('participant_id', 'size'),
        modal_frac=('participant_classification',
                    lambda s: s.value_counts(normalize=True).iloc[0]),
    ).reset_index()

    missing = [
        (r['image_name'], r['noise_level']) for _, r in grp.iterrows()
        if not os.path.exists(os.path.join(ROOT, 'Noisy Images',
                                           f"phase_noise_{int(r['noise_level']):03d}",
                                           f"{r['image_name']}.png"))
    ]
    if missing:
        print(f"WARNING: {len(missing)} images not found, e.g. {missing[:3]}")
        grp = grp[~grp.set_index(['image_name','noise_level']).index.isin(missing)].reset_index(drop=True)

    print(f"instances: {len(grp)}  |  mean annotators per instance: {grp['n_ann'].mean():.2f}")

    model = MuHRegressor(pretrained=False).to(device)
    ck = torch.load('checkpoints/mu_h_regressor.pt', map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    loader = DataLoader(IN16HDataset(grp, get_eval_transforms()),
                        batch_size=batch_size, shuffle=False, num_workers=2)
    preds = []
    with torch.no_grad():
        for b in loader:
            preds.append(model(b['image'].to(device)).cpu().numpy())
    grp['mu_hat_H'] = np.concatenate(preds)

    os.makedirs('results/tables', exist_ok=True)
    grp.to_csv('results/tables/imagenet16h_zeroshot.csv', index=False)

    print("\n=== zero-shot transfer: predicted vs elicited human confidence ===")
    pr, pp = pearsonr(grp['mu_hat_H'], grp['mu_H_true'])
    sr, sp = spearmanr(grp['mu_hat_H'], grp['mu_H_true'])
    print(f"  Pearson  r = {pr:+.4f}  (p = {pp:.2e})")
    print(f"  Spearman r = {sr:+.4f}  (p = {sp:.2e})")
    print(f"  HACM-10 in-domain reference: r = 0.507")

    print("\n=== by noise level ===")
    for lv, g in grp.groupby('noise_level'):
        r_ = pearsonr(g['mu_hat_H'], g['mu_H_true'])[0]
        print(f"  omega={lv}: n={len(g):4d}  pred {g['mu_hat_H'].mean():.4f}  "
              f"true {g['mu_H_true'].mean():.4f}  acc {g['human_acc'].mean():.4f}  r={r_:+.4f}")

    print("\n=== does the prediction track difficulty, not just confidence? ===")
    print(f"  pred vs human accuracy : r = {pearsonr(grp['mu_hat_H'], grp['human_acc'])[0]:+.4f}")
    print(f"  pred vs modal agreement: r = {pearsonr(grp['mu_hat_H'], grp['modal_frac'])[0]:+.4f}")
    print(f"  pred vs noise level    : r = {pearsonr(grp['mu_hat_H'], grp['noise_level'])[0]:+.4f}")

    return grp


if __name__ == '__main__':
    run()
