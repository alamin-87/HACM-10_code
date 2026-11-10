# HA-ECN

Training and analysis code for *The Calibration Cliff: Human–Model Confidence
Divergence under Compound Visual Ambiguity*.

HACM is the signed, instance-level difference between a model's confidence in
its predicted class and the certainty a person reports on the same image. On
HACM-10 that gap widens as visual degradations compose: mean |HACM| runs 0.05 on
undegraded images, 0.14 at one degradation and 0.19 at two, while accuracy falls
only from 0.984 to 0.959. The model stays confident where people stop being so.

This repository holds the code. The corpus and the trained checkpoints are too
large for git and are deposited separately.

## Getting the data and the models

| | Size | Licence | Where |
|---|---|---|---|
| HACM-10 corpus: 10,000 images, annotations, splits | 7.8 GB | CC BY 4.0 | [doi:10.17632/3k5dnvhzw9.1](https://doi.org/10.17632/3k5dnvhzw9.1) |
| Trained checkpoints, fitted temperatures, per-fold tables | 2.9 GB | MIT | [doi:10.17632/nbbymp9zp6.1](https://doi.org/10.17632/nbbymp9zp6.1) |
| This code, archived with a DOI | 13 MB | MIT | [doi:10.17632/xwfbms6wpn.1](https://doi.org/10.17632/xwfbms6wpn.1) |

Download both deposits and this repository into one directory, so that they sit
side by side:

```
somewhere/
|-- 1_dataset/
|-- 2_models/
+-- HACM-10_code/
```

then from that directory:

```bash
bash HACM-10_code/assemble.sh
```

`assemble.sh` creates symlinks under the repository pointing back at the two
downloads, which stay untouched. Symlinks need a filesystem that supports them;
a mounted cloud drive usually does not, and the script stops with a message if a
link fails. The table at the end of this file gives the layout for placing the
files by hand instead.

ImageNet-16H, used by the transfer analysis, is released by
[Steyvers et al. on OSF](https://osf.io/2ntrf/) and is not redistributed here.
Place it under `data/imagenet16h/` with its directory structure unchanged.

## Running it

```bash
python -m pip install -r requirements.txt
python analysis/cv_hacm.py
```

Python 3.11 or later. Every script writes to `results/tables/` and is safe to
re-run; completed folds are skipped.

| Result | Script |
|---|---|
| §7.1–7.3 calibration cliff, per-architecture slopes | `analysis/cv_hacm.py` |
| §7.2.2 calibration error against misalignment | `analysis/calibration_vs_hacm.py` |
| §7.2.3 analysis of variance and Tukey contrasts | `analysis/cv_analysis.py` |
| §7.4 joint confidence matrix | `analysis/quadrant_analysis.py` |
| §7.4 whether annotator uncertainty ranks model errors | `analysis/quadrant_error_auc.py` |
| §7.5 human-confidence regressor | `analysis/train_mu_h_regressor.py`, then `analysis/mu_h_predict.py` |
| §7.5 zero-shot transfer to ImageNet-16H | `analysis/imagenet16h_zeroshot.py` |
| §7.5 divergence ratio on ImageNet-16H | `analysis/imagenet16h_ratio.py` |
| §7.6 deferral and team accuracy | `analysis/deferral_team.py` |
| §7.6 anchoring sensitivity of the deferral gain | `analysis/anchoring_sensitivity.py` |
| §7.7 Monte Carlo dropout, conformal sets | `analysis/cv_mc_dropout.py`, `analysis/cv_raps.py` |
| §7.8 objective ablation, five folds | `analysis/run_ablation_5fold.py` |
| Figures | `analysis/figures.py` |
| The Data in Brief figures | `figures/make_dib_figures.py` |

Runtimes on a single A100: the ensemble about 17 minutes per epoch at batch size
32, the deterministic baselines proportionally less, Monte Carlo dropout at
T = 50 over the test partition about 20 minutes per fold. The ablation is 20
runs of five epochs each.

## Layout

```
datasets/      loaders, transforms, quality control, split construction
models/        backbones, ensemble, temperature scaling, MC dropout, RAPS,
               the mu_H regressor, and the Module B logistic model
losses/        the objectives compared in §7.8
trainers/      cross-validation and ablation training loops
metrics/       expected calibration error
analysis/      one script per reported analysis
figures/       the Data in Brief figures
results/       the tables, figures and statistics the analysis produced
docs/          the hand-drawn diagrams and the graphical abstract
```

| Module | Where |
|---|---|
| A, the uncertainty-aware ensemble | `models/backbone/`, `models/module_a/` |
| B, the divergence predictor | `models/module_b/` |
| C, the deferral rule | `analysis/deferral_team.py` |

Module C is a decision rule over quantities the first two produce rather than a
network, so it has no directory of its own.

`trainers/ablation_trainer.py` branches on a `config` string. Eight branches
produced reported results: `ce`, `human_ls`, `cmp` and `hybrid` for the
five-fold ablation; `hrc`, `hrc_anchored` and `dal` for the single-fold weight
sweep; and `crl`, a correctness-ranking objective reaching accuracy 0.9553 at
slope −0.0240. Two more were never run — `hca`, absolute confidence matching,
and `focal` — but their loss modules stay under `losses/` because the trainer
imports every loss at module level.

`trainers/cv_trainer.py` trains any of the four architectures: it takes a
`model_builder` and a `loss_fn` as arguments, so the ensemble and the three
baselines all run through it. Its `loss_fn` is called as
`loss_fn(outputs, labels, H_label)`. Note that the `lr_head` argument is
accepted but not used; the optimiser applies `lr_backbone` to every parameter.

## Notes on the released tables

`results/tables/ablation_5fold.csv` records `lambda` as null for `ce`, which has
no penalty term. The same placeholder survives in one filename under
`checkpoints/ablation/`: the fold-0 run is stored as `ce_fold0_lam1.0.json` with
`"lambda": 1.0` where the other four folds record 0.5. Cross-entropy takes no
penalty weight, so the value never entered the loss and the trained weights are
unaffected.

Three RAPS summaries sit under `results/tables/`. `cv_raps_summary.csv` is the
one the article reports; `cv_raps_summary_alpha002.csv` and
`cv_raps_a002_l001.csv` are earlier parameter settings, kept so the choice can
be seen rather than taken on trust. `hybrid_loss_ablation.csv` holds the fold-0
runs and is read back by `analysis/run_ablation_5fold.py` as its seed, so it is
an input rather than a leftover and deleting it will stop that script.
`ablation_fold0_first.csv` is an earlier fold-0 run that `ablation_5fold.csv`
supersedes, and `hrc_lambda_sweep.csv` holds three of the four `hrc` runs; the
fourth is in `checkpoints/ablation/`.

`analysis/cv_mc_dropout.py` and `analysis/cv_raps.py` write per-fold
intermediates to `data/processed/cv_mc_dropout/` and `data/processed/cv_raps/`.
Those intermediates are not in the deposits: the Monte Carlo columns were merged
into `cv_master_full.csv` in the model deposit, and the conformal results are
summarised in `results/tables/cv_raps_summary.csv`.

Twenty-two of the released files under `results/` were produced interactively
rather than by a script here, and the scripts will not regenerate them: both
files under `statistics/`, and under `tables/` `ablation_fold0_first`,
`annotator_attrition_check`, `cv_calibration_cliff_main`, `cv_hacm_per_fold`,
`cv_hacm_summary`, `cv_mc_vs_det`, the three `cv_module_b_*`, the three
`cv_raps_*`, `dal_ablation`, `ece_before_after`, `hrc_lambda_sweep`,
`joint_fdr`, `leave_one_annotator_out`, `slope_by_architecture`,
`team_by_architecture` and `temperature_effect_on_slope`. They are released so
that every number in the article can be read off a file, and each derives from
the per-fold model outputs released in full in the model deposit under
`cv_hacm/`, so any of them can be recomputed from material that is here.
`analysis/cv_analysis.py`, which produces the analysis of variance and the Tukey
contrasts, prints its results rather than saving them.

`analysis/figures.py` draws figures 2, 5 and 7 more than once. The later block
in each case is what the article carries; the earlier one is kept so the change
can be seen. The script runs top to bottom, so the last write for a filename is
the published figure.

Two known defects in the corpus are recorded in the corpus deposit, in
`duplicates.csv` and `label_corrections.csv`: 486 of the 10,000 image files
duplicate the bytes of another file, and seven carry a category prefix that
contradicts their content. Excluding the 103 test images whose content appears
in the training pool moves ensemble test accuracy from 0.9774 to 0.9765 and mean
|HACM| from 0.1340 to 0.1344.

## Placing the files by hand

| Path the scripts expect | Comes from |
|---|---|
| `data/images/` | `1_dataset/images/` |
| `data/train.csv` and the other three splits | `1_dataset/splits/` |
| `data/processed/annotations_qc_passed.csv` | `1_dataset/annotations/` |
| `data/annotations_pseudonymized.csv` | `1_dataset/annotations/annotations_raw.csv` |
| `data/processed/human_signals.csv` | `1_dataset/annotations/` |
| `data/processed/cv_pooled.csv` | `1_dataset/splits/` |
| `data/processed/cv_master_full.csv` | `2_models/cv_hacm/` |
| `data/processed/cv_hacm/` | `2_models/cv_hacm/` |
| `checkpoints/cv_module_a/` | `2_models/ensemble_folds/` |
| `checkpoints/cv_baselines/` | `2_models/baseline_folds/` |
| `checkpoints/ablation/` | `2_models/ablation/` |
| `checkpoints/mu_h_regressor.pt` | `2_models/` |

## Citation

The article is under review; `CITATION.cff` carries the machine-readable form.

    Al-Amin, M., Toufick, M. Y. A., Islam, S. K. M. S., & Sattar, A. (2026).
    The Calibration Cliff: Human-Model Confidence Divergence under Compound
    Visual Ambiguity. PLOS ONE. Manuscript PONE-D-26-52711, under review.

To cite this code rather than the article, use the archived deposit:

    Al-Amin, Md.; Toufick, Md Yeasir Arafat; Islam, S.K.M Shadekul;
    Sattar, Abdus (2026), "HA-ECN: analysis code for human-model confidence
    divergence under compound visual ambiguity", Mendeley Data, V1,
    doi: 10.17632/xwfbms6wpn.1

## Licence

MIT, see [LICENSE](LICENSE). The corpus is released separately under CC BY 4.0.
