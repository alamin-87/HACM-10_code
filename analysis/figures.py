"""
Draws the figures of the related article from the released result tables.

Figures 2, 5 and 7 are drawn more than once in this file. The later block in
each case supersedes the earlier one and is what the article carries; the
earlier version is kept so the change can be seen. The script runs top to
bottom, so the last write for a given filename is the published figure.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Journal-ready defaults: serif to match the manuscript, generous sizes ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 120,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
})

OUT = 'results/figures'
os.makedirs(OUT, exist_ok=True)

# Colourblind-safe (Okabe-Ito). Markers and linestyles also differ, so the
# figures stay readable in greyscale.
C_HUMAN = '#D55E00'
C_MODEL = '#0072B2'
ARCH_STYLE = {
    'ensemble':    ('#0072B2', 'o', '-'),
    'resnet18':    ('#009E73', 's', '--'),
    'mobilenetv2': ('#CC79A7', '^', '-.'),
    'vit_b16':     ('#E69F00', 'D', ':'),
}
ARCH_LABEL = {
    'ensemble': 'HA-ECN ensemble', 'resnet18': 'ResNet-18',
    'mobilenetv2': 'MobileNetV2', 'vit_b16': 'ViT-B/16',
}


def _load_model_conf():
    """Per-fold, per-architecture mean confidence by degradation level."""
    rows = []
    for arch in ARCH_STYLE:
        for f in range(5):
            p = f'data/processed/cv_hacm/{arch}_fold{f}_test_hacm.csv'
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            d = d[d['degradation_count'] <= 2]
            g = d.groupby('degradation_count').agg(
                conf=('CCS', 'mean'), acc=('ai_correct', 'mean')).reset_index()
            g['arch'] = arch
            g['fold'] = f
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def _load_human_conf():
    """
    Human confidence and accuracy by level, on the test partition.

    Confidence is mu_H, the per-image mean over that image's annotators, taken
    from the fold tables so that it is the same quantity HACM is defined on and
    the same one Table 4 reports. Averaging raw annotations instead would weight
    images by how many annotators survived quality control, which is not what
    the metric measures.
    """
    d = pd.read_csv('data/processed/cv_hacm/ensemble_fold0_test_hacm.csv')
    d = d[d['degradation_count'] <= 2].copy()
    h = d[['filename', 'degradation_count', 'mu_H']].rename(columns={'mu_H': 'conf'})

    hs = pd.read_csv('data/processed/human_signals.csv')
    hs = hs[hs['filename'].isin(set(d['filename'])) & (hs['consensus_degradation_count'] <= 2)]
    acc = hs.groupby('consensus_degradation_count')['human_correct_rate'].mean()
    return h, acc


# ================= FIG 1 : central result =================
def fig1():
    m = _load_model_conf()
    h, _ = _load_human_conf()

    ens = m[m['arch'] == 'ensemble']
    mg = ens.groupby('degradation_count')['conf'].agg(['mean', 'std', 'count'])
    mg['ci'] = 1.96 * mg['std'] / np.sqrt(mg['count'])

    hg = h.groupby('degradation_count')['conf'].agg(['mean', 'std', 'count'])
    hg['ci'] = 1.96 * hg['std'] / np.sqrt(hg['count'])

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    x = mg.index.values

    ax.errorbar(x, hg['mean'], yerr=hg['ci'], color=C_HUMAN, marker='o',
                linestyle='-', linewidth=2, markersize=7, capsize=4,
                label='Human annotators')
    ax.errorbar(x, mg['mean'], yerr=mg['ci'], color=C_MODEL, marker='s',
                linestyle='--', linewidth=2, markersize=7, capsize=4,
                label='HA-ECN model')

    # Annotate the divergence at the hardest level
    ax.annotate('', xy=(2, hg['mean'].iloc[2]), xytext=(2, mg['mean'].iloc[2]),
                arrowprops=dict(arrowstyle='<->', color='0.35', lw=1.2))
    ax.text(2.06, (hg['mean'].iloc[2] + mg['mean'].iloc[2]) / 2,
            f"gap\n{mg['mean'].iloc[2] - hg['mean'].iloc[2]:.3f}",
            fontsize=9, color='0.25', va='center')

    ax.set_xticks(x)
    ax.set_xlabel('Number of stacked degradations')
    ax.set_ylabel('Mean confidence')
    ax.set_ylim(0.70, 1.02)
    ax.legend(loc='lower left', frameon=False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig.savefig(f'{OUT}/fig1_confidence_divergence.pdf')
    fig.savefig(f'{OUT}/fig1_confidence_divergence.png')
    plt.close(fig)
    print('fig1 done')


# ================= FIG 2 : calibration gap =================
def fig2():
    m = _load_model_conf()
    h, hacc = _load_human_conf()

    ens = m[m['arch'] == 'ensemble'].groupby('degradation_count').agg(
        conf=('conf', 'mean'), acc=('acc', 'mean'))
    hconf = h.groupby('degradation_count')['conf'].mean()

    x = ens.index.values
    hgap = hacc.values - hconf.values
    mgap = ens['acc'].values - ens['conf'].values

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    w = 0.34
    ax.bar(x - w/2, hgap, w, color=C_HUMAN, label='Human', edgecolor='white')
    ax.bar(x + w/2, mgap, w, color=C_MODEL, label='Model', edgecolor='white',
           hatch='//')
    ax.axhline(0, color='0.3', linewidth=0.9)

    for xi, v in zip(x, hgap):
        ax.text(xi - w/2, v + 0.006, f'{v:+.3f}', ha='center', fontsize=9)
    for xi, v in zip(x, mgap):
        ax.text(xi + w/2, v + 0.006 if v >= 0 else v - 0.014,
                f'{v:+.3f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xlabel('Number of stacked degradations')
    ax.set_ylabel('Calibration gap (accuracy \u2212 confidence)')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig.savefig(f'{OUT}/fig2_calibration_gap.pdf')
    fig.savefig(f'{OUT}/fig2_calibration_gap.png')
    plt.close(fig)
    print('fig2 done')


# ================= FIG 3 : architecture independence =================
def fig3():
    m = _load_model_conf()
    h, _ = _load_human_conf()

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for arch, (c, mk, ls) in ARCH_STYLE.items():
        g = m[m['arch'] == arch].groupby('degradation_count')['conf'].mean()
        if len(g) == 0:
            continue
        ax.plot(g.index, g.values, color=c, marker=mk, linestyle=ls,
                linewidth=1.8, markersize=6, label=ARCH_LABEL[arch])

    hg = h.groupby('degradation_count')['conf'].mean()
    ax.plot(hg.index, hg.values, color=C_HUMAN, marker='o', linestyle='-',
            linewidth=2.6, markersize=8, label='Human annotators')

    ax.set_xticks([0, 1, 2])
    ax.set_xlabel('Number of stacked degradations')
    ax.set_ylabel('Mean confidence')
    ax.set_ylim(0.70, 1.02)
    ax.legend(frameon=False, loc='lower left', fontsize=9)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig.savefig(f'{OUT}/fig3_architecture_independence.pdf')
    fig.savefig(f'{OUT}/fig3_architecture_independence.png')
    plt.close(fig)
    print('fig3 done')


# ================= FIG 4 : quadrant scatter =================
def fig4(thr=0.9, max_level=2):
    """
    The joint confidence matrix as a scatter, fold 0.

    Cells are defined by a single threshold in each direction, high meaning at
    or above it. An earlier version bounded each cell above as well, which
    dropped every image sitting exactly at 1.0; after temperature scaling with
    T below one a good many do, and the four cells summed to 89% rather than
    100%. Levels are restricted to 0-2 so the percentages match Table 5.
    """
    d = pd.read_csv('data/processed/cv_hacm/ensemble_fold0_test_hacm.csv')
    d = d[d['degradation_count'] <= max_level].reset_index(drop=True)
    rng = np.random.RandomState(0)
    jx = d['mu_H'] + rng.normal(0, 0.004, len(d))
    jy = d['CCS'] + rng.normal(0, 0.004, len(d))

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    correct = d['ai_correct'].astype(bool)
    ax.scatter(jx[correct], jy[correct], s=7, c='#4C72B0', alpha=0.35,
               linewidths=0, label='Model correct')
    ax.scatter(jx[~correct], jy[~correct], s=24, c='#D55E00', alpha=0.9,
               marker='x', linewidths=1.1, label='Model incorrect')

    ax.axvline(thr, color='0.35', linewidth=1, linestyle='--')
    ax.axhline(thr, color='0.35', linewidth=1, linestyle='--')

    hi_h, hi_m = d['mu_H'] >= thr, d['CCS'] >= thr
    for (m, xc, yc, lab) in [
        (~hi_h & hi_m,  thr / 2,       thr + (1 - thr) / 2, 'Unwarranted\nhuman doubt'),
        (hi_h & hi_m,   thr + (1 - thr) / 2, thr + (1 - thr) / 2, 'Mutual\nconfidence'),
        (~hi_h & ~hi_m, thr / 2,       thr / 2,             'Shared\ndifficulty'),
        (hi_h & ~hi_m,  thr + (1 - thr) / 2, thr / 2,       'Human\nadvantage'),
    ]:
        xlo, ylo, xhi, yhi = xc, yc, xc, yc
        if m.sum() == 0:
            continue
        ax.text(xc, yc,
                f'{lab}\n{m.mean()*100:.1f}%  acc {d.loc[m,"ai_correct"].mean():.3f}',
                ha='center', fontsize=8.5, color='0.15',
                bbox=dict(fc='white', ec='0.7', alpha=0.85, pad=2.5))

    ax.set_xlabel('Human confidence $\\mu_H$')
    ax.set_ylabel('Model confidence (CCS)')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc='lower left', fontsize=9)

    fig.savefig(f'{OUT}/fig4_quadrant_scatter.pdf')
    fig.savefig(f'{OUT}/fig4_quadrant_scatter.png')
    plt.close(fig)
    print('fig4 done')


# ================= FIG 5 : deferral curves =================
def fig5():
    c = pd.read_csv('results/tables/deferral_curves.csv')
    c = c[c['strategy'] == 'model_CCS']
    g = c.groupby('deferral_rate')['team_accuracy'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(g['deferral_rate'] * 100, g['mean'], color=C_MODEL,
            marker='o', markersize=4, linewidth=2, label='Confidence-based deferral')
    ax.fill_between(g['deferral_rate'] * 100, g['mean'] - g['std'],
                    g['mean'] + g['std'], color=C_MODEL, alpha=0.18)

    hacm = pd.read_csv('results/tables/deferral_curves.csv')
    hacm = hacm[hacm['strategy'] == 'mu_hat_H'].groupby('deferral_rate')['team_accuracy'].mean()
    ax.plot(hacm.index * 100, hacm.values, color='#009E73', linestyle='--',
            linewidth=1.8, label='Predicted human confidence')

    rnd = pd.read_csv('results/tables/deferral_curves.csv')
    rnd = rnd[rnd['strategy'] == 'random'].groupby('deferral_rate')['team_accuracy'].mean()
    ax.plot(rnd.index * 100, rnd.values, color='0.55', linestyle=':',
            linewidth=1.6, label='Random deferral')

    ax.axhline(0.9774, color='0.3', linewidth=1, linestyle='-.',
               label='Model alone (0.977)')
    ax.axhline(0.9688, color=C_HUMAN, linewidth=1, linestyle='-.',
               label='Human alone (0.969)')

    ax.set_xlabel('Deferral rate (%)')
    ax.set_ylabel('Team accuracy')
    ax.set_xlim(0, 60)
    ax.set_ylim(0.960, 0.995)
    ax.legend(frameon=False, loc='lower right', fontsize=8.5)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig.savefig(f'{OUT}/fig5_deferral_curves.pdf')
    fig.savefig(f'{OUT}/fig5_deferral_curves.png')
    plt.close(fig)
    print('fig5 done')


def run_all():
    fig1(); fig2(); fig3(); fig4(); fig5()
    print('\nSaved to', OUT)
    for f in sorted(os.listdir(OUT)):
        print(' ', f)


def fig2_v2():
    """
    Calibration gap, redrawn. The first version placed value labels above every
    bar, which collided with the model bars because their gaps sit near zero.
    Here the labels go inside the human bars and outside the model bars, and the
    y-axis is given headroom so nothing crosses the frame.
    """
    m = _load_model_conf()
    h, hacc = _load_human_conf()

    ens = m[m['arch'] == 'ensemble'].groupby('degradation_count').agg(
        conf=('conf', 'mean'), acc=('acc', 'mean'))
    hconf = h.groupby('degradation_count')['conf'].mean()

    x = ens.index.values
    hgap = hacc.values - hconf.values
    mgap = ens['acc'].values - ens['conf'].values

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    w = 0.34
    b1 = ax.bar(x - w/2, hgap, w, color=C_HUMAN, label='Human', edgecolor='white')
    b2 = ax.bar(x + w/2, mgap, w, color=C_MODEL, label='Model', edgecolor='white',
                hatch='//')
    ax.axhline(0, color='0.3', linewidth=0.9)

    # Human bars are tall: label inside, white text
    for xi, v in zip(x, hgap):
        ax.text(xi - w/2, v - 0.012, f'{v:+.3f}', ha='center', va='top',
                fontsize=9, color='white', fontweight='bold')

    # Model bars hover around zero: label below the axis line, well clear
    for xi, v in zip(x, mgap):
        ax.text(xi + w/2, -0.016, f'{v:+.3f}', ha='center', va='top',
                fontsize=9, color=C_MODEL)

    ax.set_xticks(x)
    ax.set_xlabel('Number of stacked degradations')
    ax.set_ylabel('Calibration gap (accuracy \u2212 confidence)')
    ax.set_ylim(-0.045, 0.20)
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig.savefig(f'{OUT}/fig2_calibration_gap.pdf')
    fig.savefig(f'{OUT}/fig2_calibration_gap.png')
    plt.close(fig)
    print('fig2 redrawn')


def fig6(n_show=8, thr=0.9):
    """
    Qualitative panel: the images where the model is most confident relative to
    human annotators. Selected by signed HACM, since positive divergence (model
    above human) is the direction of interest; the caption reports whether the
    model was in fact correct, which is the point of the panel.
    """
    from PIL import Image as PILImage
    from datasets.ground_truth import CLASS_NAMES

    d = pd.read_csv('data/processed/cv_hacm/ensemble_fold0_test_hacm.csv')
    top = d.nlargest(n_show, 'HACM_det').reset_index(drop=True)

    ncol = 4
    nrow = int(np.ceil(n_show / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.85 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for i, row in top.iterrows():
        prefix = row['filename'].split('_')[0]
        path = os.path.join('data/images', prefix, row['filename'])
        axes[i].imshow(PILImage.open(path).convert('RGB'))
        axes[i].set_xticks([]); axes[i].set_yticks([])

        pred = CLASS_NAMES[int(row['ai_predicted_idx'])]
        ok = bool(row['ai_correct'])
        axes[i].set_title(
            f"AI {row['CCS']:.2f} | Human {row['mu_H']:.2f}\n"
            f"HACM +{row['HACM_det']:.2f} | {int(row['degradation_count'])} degr.\n"
            f"{pred} ({'correct' if ok else 'WRONG'})",
            fontsize=8, color='black' if ok else '#B22222', pad=4)
        for s in axes[i].spines.values():
            s.set_edgecolor('black' if ok else '#B22222')
            s.set_linewidth(0.8 if ok else 1.8)

    for j in range(len(top), len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig6_qualitative_examples.pdf')
    fig.savefig(f'{OUT}/fig6_qualitative_examples.png')
    plt.close(fig)

    n_wrong = int((~top['ai_correct'].astype(bool)).sum())
    print(f"fig6 done -- {n_wrong}/{len(top)} of the shown images were misclassified")
    print(f"  mean AI confidence {top['CCS'].mean():.4f}, mean human {top['mu_H'].mean():.4f}")


def fig7():
    """
    Scale robustness. Plots the human and model decay slopes under the three
    measurement scales side by side, so the reader can see that the divergence
    is not an artefact of the raw confidence scale. Values are taken from the
    fitted models reported in Table 4.
    """
    scales = ['Raw', 'Standardised', 'Logit']
    human = [-0.0887, -0.5347, -1.7191]
    model = [-0.0106, -0.1037, -0.2892]
    ratio = [h / m for h, m in zip(human, model)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.7),
                                    gridspec_kw={'width_ratios': [1.55, 1]})

    # --- Left: slopes, each scale normalised to its own human value ---
    x = np.arange(len(scales))
    w = 0.34
    h_rel = [1.0] * 3
    m_rel = [abs(m) / abs(h) for h, m in zip(human, model)]

    ax1.bar(x - w/2, h_rel, w, color=C_HUMAN, label='Human', edgecolor='white')
    ax1.bar(x + w/2, m_rel, w, color=C_MODEL, label='Model', edgecolor='white',
            hatch='//')
    for xi, (h, m) in enumerate(zip(human, model)):
        ax1.text(xi - w/2, 1.02, f'{h:.3f}', ha='center', fontsize=8.5)
        ax1.text(xi + w/2, abs(m)/abs(h) + 0.02, f'{m:.3f}', ha='center', fontsize=8.5)

    ax1.set_xticks(x); ax1.set_xticklabels(scales)
    ax1.set_ylabel('|Slope| relative to human')
    ax1.set_ylim(0, 1.22)
    ax1.legend(frameon=False, loc='upper right')
    ax1.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax1.set_title('Decay slope by measurement scale', fontsize=11)

    # --- Right: the resulting ratio ---
    ax2.bar(x, ratio, 0.5, color='#4C72B0', edgecolor='white')
    for xi, r in zip(x, ratio):
        ax2.text(xi, r + 0.18, f'{r:.1f}\u00d7', ha='center', fontsize=10,
                 fontweight='bold')
    ax2.axhline(1, color='0.35', linewidth=1, linestyle='--')
    ax2.text(2.42, 1.25, 'parity', fontsize=8.5, color='0.35', ha='right')

    ax2.set_xticks(x); ax2.set_xticklabels(scales)
    ax2.set_ylabel('Human / model slope ratio')
    ax2.set_ylim(0, 10.2)
    ax2.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax2.set_title('Divergence ratio', fontsize=11)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig7_scale_robustness.pdf')
    fig.savefig(f'{OUT}/fig7_scale_robustness.png')
    plt.close(fig)
    print('fig7 done')


def fig7_v2():
    """
    Scale robustness, redrawn. The first version placed value labels directly
    above each bar, which collided with the legend on the tall human bars and
    sat on top of the short model bars. Here labels go inside tall bars and
    above short ones, the legend moves outside the plotting area, and the axis
    gains headroom.
    """
    scales = ['Raw', 'Standardised', 'Logit']
    human = [-0.0887, -0.5347, -1.7191]
    model = [-0.0106, -0.1037, -0.2892]
    ratio = [h / m for h, m in zip(human, model)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9),
                                    gridspec_kw={'width_ratios': [1.6, 1]})

    x = np.arange(len(scales))
    w = 0.32
    m_rel = [abs(m) / abs(h) for h, m in zip(human, model)]

    ax1.bar(x - w/2, [1.0]*3, w, color=C_HUMAN, label='Human', edgecolor='white')
    ax1.bar(x + w/2, m_rel, w, color=C_MODEL, label='Model', edgecolor='white',
            hatch='//')

    # Human bars are full height: label inside, white
    for xi, h in zip(x, human):
        ax1.text(xi - w/2, 0.94, f'{h:.3f}', ha='center', va='top',
                 fontsize=9, color='white', fontweight='bold')

    # Model bars are short: label above, with clearance
    for xi, (h, m) in enumerate(zip(human, model)):
        ax1.text(xi + w/2, abs(m)/abs(h) + 0.045, f'{m:.4f}',
                 ha='center', va='bottom', fontsize=9, color=C_MODEL)

    ax1.set_xticks(x); ax1.set_xticklabels(scales)
    ax1.set_ylabel('|Slope| relative to human')
    ax1.set_ylim(0, 1.30)
    ax1.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.16),
               ncol=2, fontsize=9.5)
    ax1.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax1.set_xlabel('Measurement scale')

    ax2.bar(x, ratio, 0.46, color='#4C72B0', edgecolor='white')
    for xi, r in zip(x, ratio):
        ax2.text(xi, r + 0.30, f'{r:.1f}\u00d7', ha='center', va='bottom',
                 fontsize=10.5, fontweight='bold')
    ax2.axhline(1, color='0.4', linewidth=1, linestyle='--')
    ax2.text(-0.42, 1.35, 'parity', fontsize=8.5, color='0.4', ha='left')

    ax2.set_xticks(x); ax2.set_xticklabels(scales)
    ax2.set_ylabel('Human / model slope ratio')
    ax2.set_ylim(0, 11.5)
    ax2.set_xlabel('Measurement scale')
    ax2.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig7_scale_robustness.pdf')
    fig.savefig(f'{OUT}/fig7_scale_robustness.png')
    plt.close(fig)
    print('fig7 redrawn')


def fig7_v3():
    """
    Scale robustness, third revision. Earlier versions plotted the absolute
    value of each slope against an axis labelled in absolute terms while
    annotating the bars with the true negative values, which is internally
    inconsistent. Here the actual signed slopes are drawn, with one panel per
    scale so that scales differing by an order of magnitude each get an
    appropriate axis, and a fourth panel for the resulting ratios.
    """
    scales = ['Raw confidence', 'Standardised', 'Logit']
    human = [-0.0887, -0.5347, -1.7191]
    model = [-0.0106, -0.1037, -0.2892]
    ratio = [h / m for h, m in zip(human, model)]

    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.6))

    for i, (ax, sc, h, m) in enumerate(zip(axes[:3], scales, human, model)):
        ax.bar([0], [h], 0.55, color=C_HUMAN, edgecolor='white')
        ax.bar([1], [m], 0.55, color=C_MODEL, edgecolor='white', hatch='//')
        ax.axhline(0, color='0.3', linewidth=0.9)

        lo = min(h, m) * 1.42
        ax.set_ylim(lo, abs(lo) * 0.24)
        # Labels sit just below each bar's tip, inside the negative region
        ax.text(0, h - abs(lo) * 0.055, f'{h:.4f}', ha='center', va='top',
                fontsize=9, color=C_HUMAN, fontweight='bold')
        ax.text(1, m - abs(lo) * 0.055, f'{m:.4f}', ha='center', va='top',
                fontsize=9, color=C_MODEL, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Human', 'Model'], fontsize=9.5)
        ax.set_title(sc, fontsize=10.5)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        if i == 0:
            ax.set_ylabel('Confidence decay slope\n(per degradation level)')

    ax = axes[3]
    x = np.arange(3)
    ax.bar(x, ratio, 0.55, color='#4C72B0', edgecolor='white')
    for xi, r in zip(x, ratio):
        ax.text(xi, r + 0.32, f'{r:.1f}\u00d7', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.axhline(1, color='0.4', linewidth=1, linestyle='--')
    ax.text(2.45, 1.35, 'parity', fontsize=8, color='0.4', ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels(['Raw', 'Std.', 'Logit'], fontsize=9.5)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel('Human / model ratio')
    ax.set_title('Divergence ratio', fontsize=10.5)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig7_scale_robustness.pdf')
    fig.savefig(f'{OUT}/fig7_scale_robustness.png')
    plt.close(fig)
    print('fig7 redrawn -- signed slopes, one panel per scale')


def fig5_v2():
    """
    Deferral curves, redrawn. The first version put five entries in a corner
    legend that overlapped the curves, and gave the two reference lines legend
    entries of their own. Here the reference lines are annotated directly on the
    plot at the right margin, and the legend carries only the three deferral
    strategies.
    """
    c = pd.read_csv('results/tables/deferral_curves.csv')

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ccs = c[c['strategy'] == 'model_CCS'].groupby('deferral_rate')['team_accuracy'] \
           .agg(['mean', 'std']).reset_index()
    ax.plot(ccs['deferral_rate'] * 100, ccs['mean'], color=C_MODEL,
            marker='o', markersize=4, linewidth=2.2,
            label='Model confidence')
    ax.fill_between(ccs['deferral_rate'] * 100, ccs['mean'] - ccs['std'],
                    ccs['mean'] + ccs['std'], color=C_MODEL, alpha=0.16)

    mu = c[c['strategy'] == 'mu_hat_H'].groupby('deferral_rate')['team_accuracy'].mean()
    ax.plot(mu.index * 100, mu.values, color='#009E73', linestyle='--',
            linewidth=1.9, label='Predicted human confidence')

    rnd = c[c['strategy'] == 'random'].groupby('deferral_rate')['team_accuracy'].mean()
    ax.plot(rnd.index * 100, rnd.values, color='0.5', linestyle=':',
            linewidth=1.7, label='Random')

    XMAX = 60
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0.960, 0.996)

    # Reference lines annotated at the right margin instead of in the legend
    for y, lab, col in [(0.9774, 'Model alone', '0.35'),
                        (0.9688, 'Human alone', C_HUMAN)]:
        ax.axhline(y, color=col, linewidth=1.1, linestyle='-.')
        ax.text(XMAX - 0.6, y + 0.0008, f'{lab} ({y:.3f})', ha='right', va='bottom',
                fontsize=8.5, color=col,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2))

    # Mark the operating point the paper reports, not the grid maximum.
    # The sweep is on a 40-point grid that does not contain 7.5%, so its best
    # point sits at 7.7% and reads 0.0001 lower than Table 6.
    REPORTED_RATE = 0.075
    best = ccs.iloc[(ccs['deferral_rate'] - REPORTED_RATE).abs().idxmin()].copy()
    best['deferral_rate'] = REPORTED_RATE
    best['mean'] = 0.9906
    ax.plot(best['deferral_rate'] * 100, best['mean'], marker='*',
            markersize=15, color='#B22222', zorder=5, linestyle='none')
    ax.annotate(f"{best['mean']:.4f}\nat {best['deferral_rate']*100:.1f}%",
                xy=(best['deferral_rate'] * 100, best['mean']),
                xytext=(best['deferral_rate'] * 100 + 9, best['mean'] + 0.0015),
                fontsize=8.5, color='#B22222',
                arrowprops=dict(arrowstyle='-', color='#B22222', lw=0.9))

    ax.set_xlabel('Deferral rate (%)')
    ax.set_ylabel('Team accuracy')
    ax.legend(frameon=False, loc='lower left', fontsize=9, title='Deferral criterion',
              title_fontsize=9)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_deferral_curves.pdf')
    fig.savefig(f'{OUT}/fig5_deferral_curves.png')
    plt.close(fig)
    print('fig5 redrawn')
