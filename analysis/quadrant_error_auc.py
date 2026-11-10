import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


CCS_THRESHOLD = 0.9
MU_H_THRESHOLD = 0.9


def error_rankability(fold, hacm_dir='data/processed/cv_hacm', max_level=2):
    """
    Within the cell where the model is confident and the annotators are not,
    asks whether annotator uncertainty carries any information about where the
    model is wrong.

    Two statistics are returned. The point-biserial correlation between mu_H
    and model error measures association; the AUC of mu_H as a ranker of those
    errors measures whether a reviewer working from their own hesitation could
    order the images so that the model's mistakes come first. An AUC near 0.5
    means they could not.

    The cell holds very few errors, so both estimates are imprecise on any one
    fold. What the five folds support is the absence of a systematic signal,
    not a precise value.
    """
    path = os.path.join(hacm_dir, f'ensemble_fold{fold}_test_hacm.csv')
    d = pd.read_csv(path)
    d = d[d['degradation_count'] <= max_level]

    cell = d[(d['CCS'] >= CCS_THRESHOLD) & (d['mu_H'] < MU_H_THRESHOLD)]
    err = 1 - cell['ai_correct'].astype(int)

    if err.sum() == 0:
        return dict(fold=fold, n=len(cell), n_errors=0, corr=np.nan, corr_p=np.nan, auc=np.nan)

    corr, corr_p = stats.pointbiserialr(err, cell['mu_H'])
    auc = roc_auc_score(err, -cell['mu_H'])
    return dict(fold=fold, n=len(cell), n_errors=int(err.sum()),
                corr=corr, corr_p=corr_p, auc=auc)


if __name__ == '__main__':
    rows = [error_rankability(f) for f in range(5)]
    df = pd.DataFrame(rows)

    os.makedirs('results/tables', exist_ok=True)
    df.to_csv('results/tables/quadrant_error_auc.csv', index=False)

    print("Within the high-model, low-human cell")
    for r in rows:
        print(f"  fold {r['fold']}: n={r['n']:5d}  errors={r['n_errors']:3d}  "
              f"corr={r['corr']:+.4f}  AUC={r['auc']:.3f}")

    ok = df.dropna(subset=['corr'])
    print(f"\n  mean correlation {ok['corr'].mean():+.4f}")
    print(f"  mean AUC         {ok['auc'].mean():.3f}")
    print(f"  AUC range        {ok['auc'].min():.2f} to {ok['auc'].max():.2f}")
    print(f"  errors per fold  {ok['n_errors'].min()} to {ok['n_errors'].max()} "
          f"among roughly {int(ok['n'].mean())} images")
