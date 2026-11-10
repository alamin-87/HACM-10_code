import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def load_cv_hacm(model_name='ensemble', n_folds=5, hacm_dir='data/processed/cv_hacm'):
    dfs = []
    for fold in range(n_folds):
        path = os.path.join(hacm_dir, f'{model_name}_fold{fold}_test_hacm.csv')
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        d = pd.read_csv(path)
        d['fold'] = fold
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def calibration_cliff_per_fold(df, max_level=3):
    rows = []
    for fold in sorted(df['fold'].unique()):
        f = df[df['fold'] == fold]
        for level in sorted(f['degradation_count'].unique()):
            if level > max_level:
                continue
            sub = f[f['degradation_count'] == level]
            rows.append({
                'fold': fold, 'degradation_count': level, 'n': len(sub),
                'accuracy': sub['ai_correct'].mean(),
                'mean_abs_HACM': sub['HACM_det'].abs().mean(),
                'mean_signed_HACM': sub['HACM_det'].mean(),
                'mean_CCS': sub['CCS'].mean(),
                'mean_mu_H': sub['mu_H'].mean(),
            })
    return pd.DataFrame(rows)


def summarize_cliff(cliff_per_fold):
    agg = cliff_per_fold.groupby('degradation_count').agg(
        n_mean=('n', 'mean'),
        acc_mean=('accuracy', 'mean'), acc_std=('accuracy', 'std'),
        hacm_mean=('mean_abs_HACM', 'mean'), hacm_std=('mean_abs_HACM', 'std'),
        ccs_mean=('mean_CCS', 'mean'), muH_mean=('mean_mu_H', 'mean'),
    ).reset_index()
    base_acc = agg.loc[agg['degradation_count'] == 0, 'acc_mean'].values[0]
    base_hacm = agg.loc[agg['degradation_count'] == 0, 'hacm_mean'].values[0]
    agg['acc_relative'] = agg['acc_mean'] / base_acc
    agg['hacm_relative'] = agg['hacm_mean'] / base_hacm
    agg['error_rate'] = 1 - agg['acc_mean']
    base_err = agg.loc[agg['degradation_count'] == 0, 'error_rate'].values[0]
    agg['error_relative'] = agg['error_rate'] / base_err if base_err > 0 else np.nan
    return agg


def anova_tukey_by_level(df, max_level=3, value_col='HACM_det'):
    d = df[df['degradation_count'] <= max_level].copy()
    d['abs_val'] = d[value_col].abs()
    levels = sorted(d['degradation_count'].unique())
    groups = [d[d['degradation_count'] == lvl]['abs_val'].values for lvl in levels]
    f_stat, p_val = f_oneway(*groups)
    tukey = pairwise_tukeyhsd(endog=d['abs_val'].values,
                               groups=d['degradation_count'].values, alpha=0.05)
    stats = {
        'levels': [int(l) for l in levels],
        'group_means': {int(l): float(g.mean()) for l, g in zip(levels, groups)},
        'group_ns': {int(l): int(len(g)) for l, g in zip(levels, groups)},
        'anova_F': float(f_stat), 'anova_p': float(p_val),
    }
    return stats, tukey


def one_row_per_image(df, max_level=2, value_col='HACM_det'):
    """
    Collapses the five folds to one row per image before testing.

    The folds share a test partition, so pooling their rows would count every
    image five times and shrink the standard errors accordingly. Averaging
    |HACM| across folds first keeps the unit of analysis the image, which is
    what Section 7.2.3 reports.
    """
    d = df[df['degradation_count'] <= max_level].copy()
    d['abs_val'] = d[value_col].abs()
    return d.groupby(['filename', 'degradation_count'])['abs_val'].mean().reset_index()


if __name__ == '__main__':
    df = load_cv_hacm('ensemble')

    cliff = calibration_cliff_per_fold(df, max_level=2)
    agg = summarize_cliff(cliff)
    print("Mean |HACM| by degradation level, over five folds")
    for _, r in agg.iterrows():
        print(f"  level {int(r['degradation_count'])}  n {int(r['n_mean']):5d}  "
              f"|HACM| {r['hacm_mean']:.4f} +/- {r['hacm_std']:.4f}  "
              f"accuracy {r['acc_mean']:.4f}")

    per_image = one_row_per_image(df)
    groups = [g['abs_val'].values for _, g in per_image.groupby('degradation_count')]
    F, p = f_oneway(*groups)
    grand = per_image['abs_val'].mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = ((per_image['abs_val'] - grand) ** 2).sum()

    print(f"\nAnalysis of variance on one row per image")
    print(f"  F({len(groups)-1}, {len(per_image)-len(groups)}) = {F:.2f}   p = {p:.2e}   "
          f"eta squared = {ss_b/ss_t:.4f}")
    print(f"  group sizes {[len(g) for g in groups]}")

    print(f"\n{pairwise_tukeyhsd(per_image['abs_val'].values, per_image['degradation_count'].values, alpha=0.05)}")
