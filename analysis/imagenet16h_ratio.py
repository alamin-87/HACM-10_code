"""
Section 7.5 — recomputing the divergence ratio on ImageNet-16H.

Fits the human confidence slope on the ImageNet-16H behavioural data and the
model slope for each released classifier at three levels of adaptation, then
reports the ratio. Also checks how much the ratio depends on the mapping from
the three-point confidence scale to the unit interval, and computes the human
overshoot on that corpus.

Reported in Section 7.5 and Table A8. Requires the ImageNet-16H release from
OSF, which is not redistributed with this repository.

Writes results/tables/imagenet16h_ratio.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = 'data/imagenet16h/2ntrf-osfstorage-archive'
OUT = 'results/tables/imagenet16h_ratio.csv'

CLASSES = ['knife', 'keyboard', 'elephant', 'bicycle', 'airplane', 'clock',
           'oven', 'chair', 'bear', 'boat', 'cat', 'bottle', 'truck', 'car',
           'bird', 'dog']

LEVEL = {80: 0, 95: 1, 110: 2, 125: 3}
MAPPING = {'low': 0.25, 'medium': 0.60, 'high': 0.90}

ALT_MAPPINGS = {
    'ours 0.25/0.60/0.90':  {'low': 0.25, 'medium': 0.60, 'high': 0.90},
    'even 0.17/0.50/0.83':  {'low': 1/6,  'medium': 0.50, 'high': 5/6},
    'wide 0.10/0.50/0.95':  {'low': 0.10, 'medium': 0.50, 'high': 0.95},
    'ordinal 0/0.5/1':      {'low': 0.0,  'medium': 0.50, 'high': 1.0},
}

ADAPTATION = {'baseline': 'none', 'epoch01': '1 epoch', 'epoch10': '10 epochs'}

HACM10_HUMAN_SLOPE = -0.0887
HACM10_MODEL_SLOPE = -0.0106


def human_frame():
    p = f'{ROOT}/Behavioral Data/human_only_classification_6per_img_export.csv'
    d = pd.read_csv(p)
    d['lvl'] = d['noise_level'].map(LEVEL)
    return d.dropna(subset=['lvl'])


def slope(df, value_col):
    g = df.groupby(['image_name', 'lvl'])[value_col].mean().reset_index()
    return np.polyfit(g['lvl'], g[value_col], 1)[0]


def main():
    if not os.path.isdir(ROOT):
        print(f'  {ROOT} not found. Download ImageNet-16H from OSF first.')
        return

    beh = human_frame()
    beh['conf'] = beh['confidence'].map(MAPPING)
    h_slope = slope(beh, 'conf')

    rows = []
    for tag, label in ADAPTATION.items():
        p = f'{ROOT}/Machine Classifier Predictions/hai_{tag}_model_preds_max_normalized.csv'
        if not os.path.exists(p):
            print(f'  missing {p}')
            continue
        d = pd.read_csv(p)
        d = d[d['noise_level'].isin(LEVEL)].copy()
        d['conf'] = d[CLASSES].max(axis=1)
        d['lvl'] = d['noise_level'].map(LEVEL)

        for name, g in d.groupby('model_name'):
            m_slope = np.polyfit(g['lvl'], g['conf'], 1)[0]
            rows.append(dict(classifier=name, adaptation=label,
                             accuracy=g['correct'].mean(),
                             model_slope=m_slope,
                             human_slope=h_slope,
                             ratio=abs(h_slope / m_slope)))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f'\nHuman slope on ImageNet-16H: {h_slope:+.4f}   '
          f'(HACM-10: {HACM10_HUMAN_SLOPE:+.4f})\n')
    print(f'  {"classifier":14s} {"adaptation":11s} {"acc":>6s} {"slope":>9s} {"ratio":>7s}')
    for r in rows:
        print(f'  {r["classifier"]:14s} {r["adaptation"]:11s} {r["accuracy"]:6.3f} '
              f'{r["model_slope"]:+9.4f} {r["ratio"]:7.2f}')

    r_p, p_p = stats.pearsonr(df['accuracy'], df['ratio'])
    r_s, _ = stats.spearmanr(df['accuracy'], df['ratio'])
    print(f'\n  ratio against accuracy over {len(df)} points: '
          f'Pearson {r_p:.3f} (p = {p_p:.1e}), Spearman {r_s:.3f}')
    print('  Those points are five architectures at three adaptation levels,')
    print('  so they are not independent. Within each level:')
    for label in ADAPTATION.values():
        sub = df[df['adaptation'] == label]
        if len(sub) > 2:
            r_w, _ = stats.pearsonr(sub['accuracy'], sub['ratio'])
            print(f'    {label:11s} r = {r_w:+.2f}  (n = {len(sub)})')
    gm = df.groupby('adaptation')[['accuracy', 'ratio']].mean()
    if len(gm) > 2:
        r_g, _ = stats.pearsonr(gm['accuracy'], gm['ratio'])
        print(f'    group means r = {r_g:+.2f}  (n = {len(gm)})')

    print('\n  Confidence-mapping sensitivity, against the most adapted set:')
    ten = df[df['adaptation'] == '10 epochs']
    m_mean = ten['model_slope'].mean() if len(ten) else np.nan
    for name, mp in ALT_MAPPINGS.items():
        beh['c'] = beh['confidence'].map(mp)
        s = slope(beh, 'c')
        print(f'    {name:24s} human slope {s:+.4f}   ratio {abs(s/m_mean):5.2f}')

    print('\n  Human overshoot on ImageNet-16H:')
    hb = beh.groupby('lvl').agg(conf=('conf', 'mean'), acc=('correct', 'mean'))
    dc = hb['conf'].iloc[0] - hb['conf'].iloc[-1]
    da = hb['acc'].iloc[0] - hb['acc'].iloc[-1]
    print(f'    confidence {hb["conf"].iloc[0]:.3f} -> {hb["conf"].iloc[-1]:.3f}')
    print(f'    accuracy   {hb["acc"].iloc[0]:.3f} -> {hb["acc"].iloc[-1]:.3f}')
    print(f'    overshoot  {dc/da:.2f}   (HACM-10: 3.25)')

    print(f'\n  written to {OUT}')
    return df


if __name__ == '__main__':
    main()
