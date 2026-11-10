"""
The Data in Brief figures, drawn at Elsevier's double-column width so that
nothing is scaled down in production and every label stays legible.

Changes from the first version:
  - every figure is 7.48 in (190 mm) wide, the double-column measure
  - Fig. 1 drops the category panel: ten bars all at exactly 1,000 carry no
    information, and Table 2 already states the counts
  - font sizes raised slightly, since the canvas is no longer shrunk

Run from the HA-ECN project root; writes to results/figures_dib/.
"""
import os, glob, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------- find the project root
for _root in ('/content/drive/MyDrive/HA-ECN',
              '/content/drive/MyDrive/HACM-10_release/1_dataset',
              '.'):
    if os.path.exists(os.path.join(_root, 'data/processed/human_signals.csv')):
        os.chdir(_root)
        break
else:
    hits = glob.glob('/content/drive/MyDrive/**/human_signals.csv', recursive=True)
    if not hits:
        raise SystemExit('human_signals.csv not found under MyDrive; '
                         'os.chdir() to the project root first')
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(hits[0]))))
print(f'working in {os.getcwd()}')

W2 = 7.48          # 190 mm, double column
W15 = 5.51         # 140 mm, 1.5 column

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 120, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
})

OUT = 'results/figures_dib'
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN, PINK, YELLOW = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00'

hs = pd.read_csv('data/processed/human_signals.csv')
qc = pd.read_csv('data/processed/annotations_qc_passed.csv')

IMG = next((d for d in ('data/images', 'images') if os.path.isdir(d)), None)
print(f'images in {IMG}' if IMG else 'image folder not found; fig3 will be skipped')

NAMES = {'CA': 'Calculator', 'CR': 'Charger', 'HE': 'Headset', 'KE': 'Keyboard',
         'MU': 'Mouse', 'PH': 'Phone', 'PN': 'Pen', 'RE': 'Remote',
         'WL': 'Wallet', 'WT': 'Watch'}


def fig1():
    """Two panels: how ambiguity is distributed, by count and by type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 2.9))

    lv = hs['consensus_degradation_count'].value_counts().sort_index()
    ax1.bar(lv.index, lv.values, color=ORANGE, width=0.64)
    ax1.set_xlabel('Consensus degradation count')
    ax1.set_ylabel('Images')
    ax1.set_xticks(lv.index)
    ax1.set_title('(a) Degradation count', loc='left', fontweight='bold')
    for x, v in zip(lv.index, lv.values):
        ax1.text(x, v + 95, f'{v:,}', ha='center', fontsize=8)
    ax1.set_ylim(0, lv.max() * 1.17)
    ax1.grid(axis='y', alpha=0.25, linewidth=0.6)

    cc = hs['consensus_condition'].astype(str).str.lower()
    types = [('Background\nclutter', 'clutter'), ('Blur', 'blur'),
             ('Occlusion', 'occl'), ('Low\nillumination', 'illum')]
    vals = [int(cc.str.contains(k).sum()) for _, k in types]
    ax2.bar(range(len(types)), vals, color=GREEN, width=0.6)
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels([n for n, _ in types], fontsize=8)
    ax2.set_ylabel('Images')
    ax2.set_title('(b) Condition type', loc='left', fontweight='bold')
    for i, v in enumerate(vals):
        ax2.text(i, v + 60, f'{v:,}', ha='center', fontsize=8)
    ax2.set_ylim(0, max(vals) * 1.18)
    ax2.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/dib_fig1_composition.{ext}')
    plt.close(fig)
    print(f'  dib_fig1_composition   {W2:.2f} x 2.90 in')


def fig3():
    """Eight example images, one clean, four single, three compound."""
    from PIL import Image

    def pick(cond, count):
        s = hs[(hs['consensus_condition'] == cond) &
               (hs['consensus_degradation_count'] == count)]
        if len(s) == 0:
            s = hs[hs['consensus_condition'] == cond]
        if len(s) == 0:
            return None
        s = s.sort_values('mu_H')
        return s.iloc[len(s) // 2]

    wanted = [('Baseline', 0), ('Blur', 1), ('Occlusion', 1),
              ('Low Illumination', 1), ('Background Clutter', 1),
              ('Blur, Occlusion', 2), ('Background Clutter, Low Illumination', 2),
              ('Blur, Low Illumination, Occlusion', 3)]
    rows = [(c, pick(c, n)) for c, n in wanted]
    rows = [(c, r) for c, r in rows if r is not None]

    if IMG is None:
        raise SystemExit('no image folder')
    fig, axes = plt.subplots(2, 4, figsize=(W2, 4.8))
    for ax, (cond, r) in zip(axes.ravel(), rows):
        fn = r['filename']
        p = f"{IMG}/{fn[:2].upper()}/{fn}"
        if not os.path.exists(p):
            hits = glob.glob(f'{IMG}/*/{fn}') + glob.glob(f'{IMG}/{fn}')
            p = hits[0] if hits else None
        if p:
            im = Image.open(p).convert('RGB')
            im.thumbnail((600, 600))
            ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color('0.75'); sp.set_linewidth(0.7)
        # no math in the panel: a subscript renders at 70% and falls below 7 pt
        cat = NAMES.get(r['true_label'][:2].upper(), r['true_label'])
        ax.set_title(f"{textwrap.fill(f'{cat} · {cond}', 30)}\n"
                     f"mean confidence {r['mu_H']:.2f}",
                     fontsize=8.5, pad=4)
    for ax in axes.ravel()[len(rows):]:
        ax.axis('off')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/dib_fig3_examples.{ext}')
    plt.close(fig)
    print(f'  dib_fig3_examples      {W2:.2f} x 4.80 in')


def fig4():
    """The ceiling mass, and the shift of mu_H with degradation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 2.9))

    ax1.hist(qc['confidence'], bins=np.arange(-0.5, 101.5, 1), color=BLUE,
             edgecolor='none')
    n100 = int((qc['confidence'] == 100).sum())
    ax1.annotate(f'{n100:,} ratings\nat exactly 100', xy=(100, n100),
                 xytext=(66, n100 * 0.76), fontsize=8, ha='center',
                 arrowprops=dict(arrowstyle='->', color='0.35', lw=1.0))
    ax1.set_xlabel('Elicited confidence (0–100)')
    ax1.set_ylabel('Annotations')
    ax1.set_xlim(-1, 103)
    ax1.set_title('(a) All 27,456 ratings', loc='left', fontweight='bold')
    ax1.grid(axis='y', alpha=0.25, linewidth=0.6)

    cols = [BLUE, ORANGE, GREEN, PINK]
    for i, k in enumerate(sorted(hs['consensus_degradation_count'].unique())):
        g = hs[hs['consensus_degradation_count'] == k]['mu_H']
        if len(g) < 20:
            continue
        ax2.hist(g, bins=np.arange(0, 1.02, 0.025), histtype='step',
                 linewidth=1.6, color=cols[i % len(cols)], density=True,
                 label=f'level {int(k)}  (n = {len(g):,})')
    ax2.set_xlabel('Per-image mean human confidence')
    ax2.set_ylabel('Density')
    ax2.legend(frameon=False, loc='upper left', fontsize=7.5)
    ax2.set_title('(b) By degradation count', loc='left', fontweight='bold')
    ax2.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/dib_fig4_confidence.{ext}')
    plt.close(fig)
    print(f'  dib_fig4_confidence    {W2:.2f} x 2.90 in')


if __name__ == '__main__':
    print('drawing at 190 mm, double column, so nothing is scaled in production:')
    fig1()
    fig4()
    try:
        fig3()
    except Exception as e:
        print(f'  fig3 skipped: {e}')
    print(f'\n  written to {OUT}/   smallest text 7.5 pt at final size')
