import os
import gc
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from datasets.dataset import HACM10Dataset
from datasets.transforms import get_eval_transforms
from models.module_a.temperature_scaling import fit_temperature, apply_temperature
from metrics.ece import compute_ece


def _infer(model, loader, device, is_ensemble):
    """Collects probabilities + metadata for a loader."""
    probs, labels, metas = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            out = model(images)
            p = out if is_ensemble else torch.softmax(out, dim=1)
            probs.append(p.cpu())
            labels.append(batch['label'])
            for j in range(len(batch['filename'])):
                metas.append({
                    'filename': batch['filename'][j],
                    'mu_H': batch['mu_H'][j].item(),
                    'sigma_H': batch['sigma_H'][j].item(),
                    'H_label': batch['H_label'][j].item(),
                    'degradation_count': batch['degradation_count'][j].item(),
                    'true_label_idx': batch['label'][j].item(),
                })
            del images, out, p
            gc.collect()
    return torch.cat(probs), torch.cat(labels), pd.DataFrame(metas)


def compute_cv_hacm(model_builder, model_name, checkpoint_dir, class_names,
                     is_ensemble=False, n_folds=5, batch_size=8, seed=42,
                     device=None, output_dir='data/processed/cv_hacm'):
    """
    For each fold: load its checkpoint, fit temperature on that fold's own
    validation split, then compute HACM on the fixed held-out test set.
    Saves per-fold results and returns a summary DataFrame.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    pooled = pd.read_csv('data/processed/cv_pooled.csv')
    strat = pooled['true_label'].astype(str) + '_' + pooled['consensus_degradation_count'].astype(str)
    counts = strat.value_counts()
    rare = counts[counts < n_folds].index
    strat = strat.where(~strat.isin(rare), pooled['true_label'].astype(str) + '_rare')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pooled_ds = HACM10Dataset('data/processed/cv_pooled.csv', 'data/images',
                                get_eval_transforms(), class_names)
    test_ds = HACM10Dataset('data/test.csv', 'data/images',
                             get_eval_transforms(), class_names)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    rows = []
    for fold, (_, val_idx) in enumerate(skf.split(pooled, strat)):
        out_path = os.path.join(output_dir, f'{model_name}_fold{fold}_test_hacm.csv')
        if os.path.exists(out_path):
            print(f"  fold {fold}: already computed, loading from disk")
            df = pd.read_csv(out_path)
            with open(os.path.join(output_dir, f'{model_name}_fold{fold}_meta.json')) as f:
                meta = json.load(f)
        else:
            ckpt_path = os.path.join(checkpoint_dir, f'{model_name}_fold{fold}.pt')
            if not os.path.exists(ckpt_path):
                print(f"  fold {fold}: checkpoint missing, skipping")
                continue

            print(f"  fold {fold}: loading checkpoint and running inference...")
            model = model_builder().to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])

            val_loader = DataLoader(Subset(pooled_ds, val_idx), batch_size=batch_size,
                                     shuffle=False, num_workers=0)
            val_probs, val_labels, _ = _infer(model, val_loader, device, is_ensemble)

            ece_before = compute_ece(val_probs, val_labels)
            T = fit_temperature(val_probs, val_labels)
            ece_after = compute_ece(apply_temperature(val_probs, T), val_labels)

            test_probs, test_labels, df = _infer(model, test_loader, device, is_ensemble)
            test_cal = apply_temperature(test_probs, T)

            df['CCS'] = test_cal.max(dim=1).values.numpy()
            df['ai_predicted_idx'] = test_cal.argmax(dim=1).numpy()
            df['HACM_det'] = df['CCS'] - df['mu_H']
            df['ai_correct'] = df['true_label_idx'] == df['ai_predicted_idx']

            meta = {
                'temperature': float(T),
                'ece_val_before': float(ece_before),
                'ece_val_after': float(ece_after),
                'ece_test': float(compute_ece(test_cal, test_labels)),
            }

            df.to_csv(out_path, index=False)
            with open(os.path.join(output_dir, f'{model_name}_fold{fold}_meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)

            del model, val_loader, val_probs, test_probs, test_cal
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calibration cliff ratio for this fold
        h0 = df[df['degradation_count'] == 0]['HACM_det'].abs().mean()
        h3 = df[df['degradation_count'] == 3]['HACM_det'].abs().mean()

        rows.append({
            'model': model_name,
            'fold': fold,
            'temperature': meta['temperature'],
            'ece_test': meta['ece_test'],
            'test_accuracy': df['ai_correct'].mean(),
            'mean_abs_HACM': df['HACM_det'].abs().mean(),
            'mean_signed_HACM': df['HACM_det'].mean(),
            'HACM_level0': h0,
            'HACM_level3': h3,
            'cliff_ratio': h3 / h0 if h0 > 0 else np.nan,
        })
        print(f"    acc={rows[-1]['test_accuracy']:.4f}, "
              f"mean|HACM|={rows[-1]['mean_abs_HACM']:.4f}, "
              f"cliff={rows[-1]['cliff_ratio']:.3f}")

    return pd.DataFrame(rows)

if __name__ == '__main__':
    from models.backbone.ensemble import EnsembleBackbone
    from models.baselines.resnet18 import build_resnet18
    from models.baselines.mobilenetv2 import build_mobilenetv2
    from models.baselines.vit_b16 import build_vit_b16
    from datasets.ground_truth import CLASS_NAMES

    compute_cv_hacm(lambda: EnsembleBackbone(num_classes=len(CLASS_NAMES)),
                    'ensemble', 'checkpoints/cv_module_a', CLASS_NAMES, is_ensemble=True)

    for name, builder in [('resnet18', build_resnet18),
                          ('mobilenetv2', build_mobilenetv2),
                          ('vit_b16', build_vit_b16)]:
        compute_cv_hacm(lambda b=builder: b(num_classes=len(CLASS_NAMES)),
                        name, 'checkpoints/cv_baselines', CLASS_NAMES, is_ensemble=False)
