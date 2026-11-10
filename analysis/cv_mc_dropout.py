import os
import gc
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from datasets.dataset import HACM10Dataset
from datasets.transforms import get_eval_transforms
from models.module_a.mc_dropout import mc_dropout_predict
from models.module_a.temperature_scaling import apply_temperature


def run_cv_mc_dropout(model_builder, checkpoint_dir, class_names,
                       model_name='ensemble', n_folds=5, T_mc=50,
                       batch_size=16, device=None,
                       meta_dir='data/processed/cv_hacm',
                       output_dir='data/processed/cv_mc_dropout',
                       save_every=10):
    """
    Runs MC Dropout (T=50) on the held-out test set for every CV fold.

    Reuses each fold's temperature from the earlier cv_hacm run rather than
    refitting. Saves progress every `save_every` batches, and skips images
    already completed, so an interrupted session resumes cleanly.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}")

    test_ds = HACM10Dataset('data/test.csv', 'data/images',
                             get_eval_transforms(), class_names)

    for fold in range(n_folds):
        out_path = os.path.join(output_dir, f'{model_name}_fold{fold}_mc.csv')

        # --- Fold-level resume ---
        done = set()
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            done = set(existing['filename'].tolist())
            if len(done) >= len(test_ds):
                print(f"\n=== Fold {fold}: already complete ({len(done)} images), skipping ===")
                continue
            print(f"\n=== Fold {fold}: resuming, {len(done)} images already done ===")
        else:
            print(f"\n=== Fold {fold}: starting fresh ===")

        ckpt_path = os.path.join(checkpoint_dir, f'{model_name}_fold{fold}.pt')
        if not os.path.exists(ckpt_path):
            print(f"  checkpoint missing: {ckpt_path}, skipping fold")
            continue

        meta_path = os.path.join(meta_dir, f'{model_name}_fold{fold}_meta.json')
        if not os.path.exists(meta_path):
            print(f"  meta (temperature) missing: {meta_path}, skipping fold")
            continue
        with open(meta_path) as f:
            T = json.load(f)['temperature']
        print(f"  using temperature T={T:.4f}")

        model = model_builder().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        records = []
        for i, batch in enumerate(loader):
            filenames = batch['filename']
            if all(f in done for f in filenames):
                continue

            images = batch['image'].to(device)
            p_mc, epistemic = mc_dropout_predict(model, images, T=T_mc)
            p_mc_cal = apply_temperature(p_mc.cpu(), T)

            preds = p_mc_cal.argmax(dim=1)
            ccs_mc = p_mc_cal.max(dim=1).values

            for j in range(len(filenames)):
                if filenames[j] in done:
                    continue
                records.append({
                    'filename': filenames[j],
                    'fold': fold,
                    'CCS_MC': ccs_mc[j].item(),
                    'epistemic_uncertainty': epistemic[j].item(),
                    'ai_predicted_idx_mc': preds[j].item(),
                    'mu_H': batch['mu_H'][j].item(),
                    'degradation_count': batch['degradation_count'][j].item(),
                    'true_label_idx': batch['label'][j].item(),
                })

            if i % 10 == 0:
                print(f"    batch {i+1}/{len(loader)}")

            del images, p_mc, p_mc_cal, epistemic
            gc.collect()

            if (i + 1) % save_every == 0 and records:
                _append_save(records, out_path)
                records = []

        if records:
            _append_save(records, out_path)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Fold {fold} complete -> {out_path}")

    print("\nAll folds done.")


def _append_save(new_records, out_path):
    new_df = pd.DataFrame(new_records)
    new_df['HACM_MC'] = new_df['CCS_MC'] - new_df['mu_H']
    new_df['ai_correct_mc'] = new_df['true_label_idx'] == new_df['ai_predicted_idx_mc']

    if os.path.exists(out_path):
        combined = pd.concat([pd.read_csv(out_path), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(out_path, index=False)
    print(f"      [saved: {len(combined)} images]")

if __name__ == '__main__':
    from models.backbone.ensemble import EnsembleBackbone
    from datasets.ground_truth import CLASS_NAMES

    run_cv_mc_dropout(lambda: EnsembleBackbone(num_classes=len(CLASS_NAMES)),
                      'checkpoints/cv_module_a', CLASS_NAMES)
