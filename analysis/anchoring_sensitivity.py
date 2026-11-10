"""
Section 7.6 — how much does reviewer anchoring cost the deferral gain?

Team accuracy is linear in the probability that a reviewer presented with a
deferred case simply adopts the model's answer. This traces that line from full
independence to complete deference and reports where the routing benefit
disappears. Reported in Section 7.6 as 0.9912 - 0.0141 pi, with break-even at
pi = 0.975.

Writes results/tables/anchoring_sensitivity.csv
"""

import os
import numpy as np
import pandas as pd

ANN = 'data/processed/annotations_qc_passed.csv'
HACM_DIR = 'data/processed/cv_hacm'
OUT = 'results/tables/anchoring_sensitivity.csv'
N_FOLDS = 5
MAX_LEVEL = 2
DEFER_RATE = 0.075

MODEL_ALONE = 0.9774
HUMAN_ALONE = 0.9688


def human_accuracy_per_image():
    a = pd.read_csv(ANN)
    a['true_code'] = a['filename'].str.split('_').str[0]
    a['correct'] = (a['label'] == a['true_code']).astype(int)
    return a.groupby('filename')['correct'].mean().rename('human_acc')


def main():
    h = human_accuracy_per_image()
    retained, deferred_model, deferred_human, coverage = [], [], [], []

    for fold in range(N_FOLDS):
        path = os.path.join(HACM_DIR, f'ensemble_fold{fold}_test_hacm.csv')
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[d['degradation_count'] <= MAX_LEVEL]
        d = d.join(h, on='filename').dropna(subset=['human_acc'])

        k = int(round(DEFER_RATE * len(d)))
        d = d.sort_values('CCS')
        dfr, ret = d.iloc[:k], d.iloc[k:]

        retained.append(ret['ai_correct'].mean())
        deferred_model.append(dfr['ai_correct'].mean())
        deferred_human.append(dfr['human_acc'].mean())
        coverage.append(1 - k / len(d))

    r, dm, dh, cov = (np.mean(retained), np.mean(deferred_model),
                      np.mean(deferred_human), np.mean(coverage))

    intercept = cov * r + (1 - cov) * dh
    slope = (1 - cov) * (dm - dh)
    break_even = (MODEL_ALONE - intercept) / slope

    rows = []
    for pi in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rows.append(dict(pi=pi, team_accuracy=intercept + slope * pi))
    df = pd.DataFrame(rows)
    df['intercept'] = intercept
    df['slope'] = slope
    df['break_even_pi'] = break_even

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f'\n  retained {cov*100:.1f}% at accuracy {r:.4f}')
    print(f'  deferred {(1-cov)*100:.1f}%: model {dm:.4f}, human {dh:.4f}')
    print(f'\n  team accuracy = {intercept:.4f} {slope:+.4f} pi\n')
    print(f'  {"pi":>5s} {"team":>9s}')
    for _, x in df.iterrows():
        print(f'  {x["pi"]:5.2f} {x["team_accuracy"]:9.4f}')
    print(f'\n  model alone {MODEL_ALONE:.4f}, human alone {HUMAN_ALONE:.4f}')
    print(f'  routing stays ahead of the model until pi = {break_even:.3f}')
    print('\n  Real anchoring is partial and varies by instance rather than being')
    print('  a single acceptance probability, so this locates the loss rather')
    print('  than measuring it.')

    print(f'\n  written to {OUT}')
    return df


if __name__ == '__main__':
    main()
