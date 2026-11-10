import os
import gc
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from datasets.dataset import HACM10Dataset
from datasets.transforms import get_eval_transforms
from models.module_a.temperature_scaling import apply_temperature
from models.module_a.raps_calibration import calibrate_raps, predict_sets


def _collect_probs(model, loader, device, T, is_ensemble=True):
    """Runs deterministic inference and applies that fold's temperature."""
    probs, labels, filenames = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            out = model(images)
            p = out if is_ensemble else torch.softmax(out, dim=1)
            p_cal = apply_temperature(p.cpu(), T)
            probs.append(p_cal)
            labels.append(batch['label'])
            filenames.extend(batch['filename'])
            del images, out, p
            gc.collect()
    return torch.cat(probs), torch.cat(labels), filenames


def run_cv_raps(model_builder, checkpoint_dir, class_names,
                 model_name='ensemble', n_folds=5, alpha=0.1,
                 lambda_reg=0.1, k_reg=2, batch_size=16, device=None,
                 meta_dir='data/processed/cv_hacm',
                 output_dir='data/processed/cv_raps',
                 is_ensemble=True):
    """
    For each fold: fit the RAPS threshold q_hat on the calibration split,
    then build prediction sets on the held-out test set.
    Reuses each fold's temperature from the earlier cv_hacm run.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}")

    cal_ds = HACM10Dataset('data/calibration.csv', 'data/images',
                            get_eval_transforms(), class_names)
    test_ds = HACM10Dataset('data/test.csv', 'data/images',
                             get_eval_transforms(), class_names)

    summary_rows = []

    for fold in range(n_folds):
        out_path = os.path.join(output_dir, f'{model_name}_fold{fold}_raps.csv')
        meta_out = os.path.join(output_dir, f'{model_name}_fold{fold}_raps_meta.json')

        if os.path.exists(out_path) and os.path.exists(meta_out):
            print(f"\n=== Fold {fold}: already done, loading ===")
            df = pd.read_csv(out_path)
            with open(meta_out) as f:
                meta = json.load(f)
        else:
            ckpt_path = os.path.join(checkpoint_dir, f'{model_name}_fold{fold}.pt')
            meta_path = os.path.join(meta_dir, f'{model_name}_fold{fold}_meta.json')
            if not (os.path.exists(ckpt_path) and os.path.exists(meta_path)):
                print(f"\n=== Fold {fold}: checkpoint or meta missing, skipping ===")
                continue

            with open(meta_path) as f:
                T = json.load(f)['temperature']
            print(f"\n=== Fold {fold}: T={T:.4f} ===")

            model = model_builder().to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])

            # --- Calibration split -> q_hat ---
            cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            cal_probs, cal_labels, _ = _collect_probs(model, cal_loader, device, T, is_ensemble)
            q_hat = calibrate_raps(cal_probs, cal_labels, alpha=alpha,
                                    lambda_reg=lambda_reg, k_reg=k_reg)
            print(f"  q_hat = {q_hat:.4f}")

            # --- Test split -> prediction sets ---
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            test_probs, test_labels, test_files = _collect_probs(model, test_loader, device, T, is_ensemble)
            pred_sets, set_sizes = predict_sets(test_probs, q_hat,
                                                 lambda_reg=lambda_reg, k_reg=k_reg)

            # Empirical coverage: is the true label inside C(x)?
            covered = [int(test_labels[i].item()) in pred_sets[i] for i in range(len(pred_sets))]

            df = pd.DataFrame({
                'filename': test_files,
                'fold': fold,
                'set_size': set_sizes,
                'true_label_idx': test_labels.numpy(),
                'covered': covered,
            })

            meta = {
                'fold': fold,
                'temperature': float(T),
                'q_hat': float(q_hat),
                'alpha': alpha,
                'lambda_reg': lambda_reg,
                'k_reg': k_reg,
                'mean_set_size': float(np.mean(set_sizes)),
                'empirical_coverage': float(np.mean(covered)),
                'target_coverage': 1 - alpha,
            }

            df.to_csv(out_path, index=False)
            with open(meta_out, 'w') as f:
                json.dump(meta, f, indent=2)

            del model, cal_probs, test_probs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"  mean |C(x)| = {meta['mean_set_size']:.2f}, "
              f"coverage = {meta['empirical_coverage']:.4f} (target {meta['target_coverage']:.2f})")
        summary_rows.append(meta)

    summary = pd.DataFrame(summary_rows)
    print("\n=== RAPS summary across folds ===")
    print(summary[['fold', 'q_hat', 'mean_set_size', 'empirical_coverage']].round(4).to_string(index=False))
    print(f"\nMean set size: {summary['mean_set_size'].mean():.3f} +/- {summary['mean_set_size'].std():.3f}")
    print(f"Mean coverage: {summary['empirical_coverage'].mean():.4f} +/- {summary['empirical_coverage'].std():.4f}")

    return summary

if __name__ == '__main__':
    from models.backbone.ensemble import EnsembleBackbone
    from datasets.ground_truth import CLASS_NAMES

    run_cv_raps(lambda: EnsembleBackbone(num_classes=len(CLASS_NAMES)),
                'checkpoints/cv_module_a', CLASS_NAMES)
