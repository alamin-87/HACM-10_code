import os
import numpy as np
import pandas as pd


QUADRANT_NAMES = {
    (True, False): 'AI high / Human low',
    (True, True): 'AI high / Human high',
    (False, True): 'AI low / Human high',
    (False, False): 'AI low / Human low',
}


def quadrant_table(df, threshold):
    """
    Splits the test set into the four joint-confidence states at a fixed
    threshold applied to both sides, and reports how often the model is
    actually correct in each. A fixed cut is used rather than a median split
    because the model's median confidence sits at 0.998, which would place
    genuinely confident predictions in a nominal 'low' cell.
    """
    d = df.copy()
    d['a_high'] = d['CCS'] >= threshold
    d['h_high'] = d['mu_H'] >= threshold

    rows = []
    for (a, h), name in QUADRANT_NAMES.items():
        m = (d['a_high'] == a) & (d['h_high'] == h)
        n = int(m.sum())
        rows.append({
            'threshold': threshold,
            'quadrant': name,
            'n': n,
            'pct': 100.0 * n / len(d),
            'model_accuracy': float(d.loc[m, 'ai_correct'].mean()) if n else np.nan,
            'mean_CCS': float(d.loc[m, 'CCS'].mean()) if n else np.nan,
            'mean_mu_H': float(d.loc[m, 'mu_H'].mean()) if n else np.nan,
            'mean_HACM': float(d.loc[m, 'HACM_det'].mean()) if n else np.nan,
            'mean_degradation': float(d.loc[m, 'degradation_count'].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def run(thresholds=(0.8, 0.9), n_folds=5):
    all_rows = []
    for fold in range(n_folds):
        p = f'data/processed/cv_hacm/ensemble_fold{fold}_test_hacm.csv'
        if not os.path.exists(p):
            print(f"missing fold {fold}")
            continue
        d = pd.read_csv(p)
        for t in thresholds:
            r = quadrant_table(d, t)
            r['fold'] = fold
            all_rows.append(r)

    df = pd.concat(all_rows, ignore_index=True)
    os.makedirs('results/tables', exist_ok=True)
    df.to_csv('results/tables/quadrant_per_fold.csv', index=False)

    for t in thresholds:
        sub = df[df['threshold'] == t]
        agg = sub.groupby('quadrant').agg(
            n_mean=('n', 'mean'),
            pct_mean=('pct', 'mean'),
            pct_std=('pct', 'std'),
            acc_mean=('model_accuracy', 'mean'),
            acc_std=('model_accuracy', 'std'),
            CCS=('mean_CCS', 'mean'),
            muH=('mean_mu_H', 'mean'),
            HACM=('mean_HACM', 'mean'),
            deg=('mean_degradation', 'mean'),
        ).reindex(list(QUADRANT_NAMES.values()))

        print("=" * 78)
        print(f"QUADRANT ANALYSIS -- threshold {t} (mean over {n_folds} folds)")
        print("=" * 78)
        print(agg.round(4).to_string())
        print()

    summary = df.groupby(['threshold', 'quadrant']).agg(
        pct=('pct', 'mean'), model_accuracy=('model_accuracy', 'mean')
    ).reset_index()
    summary.to_csv('results/tables/quadrant_summary.csv', index=False)
    print("Saved: results/tables/quadrant_summary.csv")
    return df


if __name__ == '__main__':
    run()
