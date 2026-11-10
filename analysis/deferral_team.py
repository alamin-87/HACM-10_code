import os
import numpy as np
import pandas as pd


def build_team_table(fold=0):
    """
    Assembles everything needed to simulate a human-AI team on the test set:
    model correctness per image, human correctness per image, predicted human
    confidence (available at inference), and the model's own confidence.

    Human correctness comes from human_correct_rate, the fraction of the three
    annotators who identified the object correctly. Treating it as the expected
    accuracy of a single randomly-drawn annotator keeps the comparison fair:
    the team defers to one person, not to a committee.
    """
    hacm = pd.read_csv(f'data/processed/cv_hacm/ensemble_fold{fold}_test_hacm.csv')
    mu_pred = pd.read_csv('data/processed/mu_h_pred/test.csv')
    human = pd.read_csv('data/processed/human_signals.csv')

    df = hacm.merge(mu_pred[['filename', 'mu_hat_H']], on='filename')
    df = df.merge(human[['filename', 'human_correct_rate', 'majority_correct']],
                  on='filename', how='left')

    df['model_correct'] = df['ai_correct'].astype(float)
    df['human_correct'] = df['human_correct_rate'].astype(float)
    return df


def coverage_curve(df, score_col, ascending=True, n_points=40):
    """
    Sweeps the deferral rate. At each rate, the lowest-scoring images (by
    score_col) go to the human and the rest are answered by the model.
    Reports team accuracy alongside model-only and human-only accuracy.
    """
    d = df.sort_values(score_col, ascending=ascending).reset_index(drop=True)
    n = len(d)
    rows = []
    for frac in np.linspace(0.0, 1.0, n_points):
        k = int(round(frac * n))
        deferred = d.iloc[:k]
        kept = d.iloc[k:]
        team = (deferred['human_correct'].sum() + kept['model_correct'].sum()) / n
        rows.append({
            'deferral_rate': frac,
            'n_deferred': k,
            'team_accuracy': team,
            'model_acc_on_kept': kept['model_correct'].mean() if len(kept) else np.nan,
            'human_acc_on_deferred': deferred['human_correct'].mean() if k else np.nan,
        })
    return pd.DataFrame(rows)


def run(n_folds=5):
    all_curves, summary = [], []

    for fold in range(n_folds):
        p = f'data/processed/cv_hacm/ensemble_fold{fold}_test_hacm.csv'
        if not os.path.exists(p):
            print(f"missing fold {fold}")
            continue

        df = build_team_table(fold)
        model_only = df['model_correct'].mean()
        human_only = df['human_correct'].mean()

        strategies = {
            'mu_hat_H': ('mu_hat_H', True),      # defer where humans would feel unsure
            'model_CCS': ('CCS', True),          # defer where the model is unsure
            'random': ('_rand', True),
        }
        df['_rand'] = np.random.RandomState(fold).rand(len(df))

        for name, (col, asc) in strategies.items():
            c = coverage_curve(df, col, ascending=asc)
            c['strategy'] = name
            c['fold'] = fold
            all_curves.append(c)

            best = c.loc[c['team_accuracy'].idxmax()]
            at20 = c.iloc[(c['deferral_rate'] - 0.20).abs().idxmin()]
            summary.append({
                'fold': fold, 'strategy': name,
                'model_only': model_only, 'human_only': human_only,
                'best_team_acc': best['team_accuracy'],
                'best_deferral_rate': best['deferral_rate'],
                'team_acc_at_20pct': at20['team_accuracy'],
            })

    curves = pd.concat(all_curves, ignore_index=True)
    summ = pd.DataFrame(summary)

    os.makedirs('results/tables', exist_ok=True)
    curves.to_csv('results/tables/deferral_curves.csv', index=False)
    summ.to_csv('results/tables/deferral_summary.csv', index=False)

    print("=" * 62)
    print("HUMAN-AI TEAM PERFORMANCE (5 folds)")
    print("=" * 62)
    print(f"  Model alone : {summ['model_only'].mean():.4f} +/- {summ['model_only'].std():.4f}")
    print(f"  Human alone : {summ['human_only'].mean():.4f} +/- {summ['human_only'].std():.4f}")
    print()
    for s in summ['strategy'].unique():
        g = summ[summ['strategy'] == s]
        print(f"  {s:10s}: best team {g['best_team_acc'].mean():.4f} "
              f"(at {g['best_deferral_rate'].mean()*100:.0f}% deferral) | "
              f"at 20% deferral {g['team_acc_at_20pct'].mean():.4f}")

    print("\n  Mean team accuracy by deferral rate:")
    piv = curves.pivot_table(index='deferral_rate', columns='strategy',
                              values='team_accuracy', aggfunc='mean')
    print(piv.iloc[::5].round(4).to_string())
    return curves, summ


def team_at_rate(rate=0.075, n_folds=5, score_col='CCS'):
    """
    Team accuracy at one deferral rate, averaged over folds.

    coverage_curve sweeps a grid, and the rate the paper reports does not fall
    on it. This evaluates the single point directly: the lowest-scoring share
    of the test set goes to a human, the rest to the model, and the two are
    scored against the true label.
    """
    accs = []
    for fold in range(n_folds):
        d = build_team_table(fold).sort_values(score_col)
        k = int(round(rate * len(d)))
        deferred, kept = d.iloc[:k], d.iloc[k:]
        accs.append((kept['ai_correct'].sum() + deferred['human_correct'].sum()) / len(d))
    return float(np.mean(accs)), float(np.std(accs, ddof=1)), accs


if __name__ == '__main__':
    run()

    mean, sd, per_fold = team_at_rate(0.075)
    print("\nAt the 7.5% deferral rate Table 6 reports 0.9906. This run gives:")
    for i, a in enumerate(per_fold):
        print(f"  fold {i}  {a:.4f}")
    print(f"  mean  {mean:.4f} +/- {sd:.4f}")
