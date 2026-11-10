import os
import numpy as np
import pandas as pd


ARCHS = ['ensemble', 'resnet18', 'mobilenetv2', 'vit_b16']


def calibration_and_misalignment(arch, hacm_dir='data/processed/cv_hacm', max_level=2):
    """
    Two quantities per architecture, averaged over folds: mean absolute
    calibration residual, and mean |HACM|.

    The residual is |CCS - correct| per image, which is not ECE. It is used here
    only as a per-architecture summary of how far stated confidence sits from
    outcome; the binned ECE for each fold is in the meta files alongside the
    HACM tables.

    The comparison the paper draws is between the two spans, not between the
    two levels: calibration quality varies considerably across these four
    models while their misalignment with human confidence does not.
    """
    cal, hacm = [], []
    for fold in range(5):
        path = os.path.join(hacm_dir, f'{arch}_fold{fold}_test_hacm.csv')
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[d['degradation_count'] <= max_level]
        cal.append((d['CCS'] - d['ai_correct'].astype(float)).abs().mean())
        hacm.append(d['HACM_det'].abs().mean())
    return dict(architecture=arch, n_folds=len(cal),
                mean_cal_err=np.mean(cal), sd_cal_err=np.std(cal, ddof=1),
                mean_abs_hacm=np.mean(hacm), sd_abs_hacm=np.std(hacm, ddof=1))


rows = [calibration_and_misalignment(a) for a in ARCHS]
df = pd.DataFrame(rows)

os.makedirs('results/tables', exist_ok=True)
df.to_csv('results/tables/calibration_vs_hacm.csv', index=False)

print("Across architectures")
for r in rows:
    print(f"  {r['architecture']:14s} cal err {r['mean_cal_err']:.4f}   "
          f"|HACM| {r['mean_abs_hacm']:.4f}")

ce, ah = df['mean_cal_err'], df['mean_abs_hacm']
print(f"\n  calibration error spans {ce.min():.4f} to {ce.max():.4f}, "
      f"a factor of {ce.max()/ce.min():.2f}")
print(f"  mean |HACM| spans      {ah.min():.4f} to {ah.max():.4f}, "
      f"a factor of {ah.max()/ah.min():.2f}")
