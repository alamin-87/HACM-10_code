#!/usr/bin/env bash
#
# Builds a working tree from the three deposits.
#
# The scripts in this repository expect the corpus and the checkpoints beside
# them, in the layout the original repository used. The deposits are split by
# licence and by how often each changes, so the paths differ; this puts them
# back together with symlinks, leaving the downloads untouched.
#
# Run from the directory holding all three deposits:
#
#     bash 3_code/assemble.sh
#
# Symlinks need a filesystem that supports them. A mounted cloud drive usually
# does not; on one, copy the files into the same layout by hand instead. The
# README gives the table.

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
DATA="$ROOT/1_dataset"
MODELS="$ROOT/2_models"

for d in "$DATA" "$MODELS"; do
    if [ ! -d "$d" ]; then
        echo "not found: $d"
        echo "download all three deposits into the same directory first."
        exit 1
    fi
done

cd "$HERE"
mkdir -p data/processed checkpoints

link () {
    if [ -e "$2" ] && [ ! -L "$2" ]; then
        echo "  skipping $2, a real file is already there"
        return
    fi
    rm -f "$2"
    if ln -s "$1" "$2" 2>/dev/null; then
        echo "  $2"
    else
        echo "  FAILED $2 -- this filesystem does not support symlinks"
        echo "         copy the files by hand; the README gives the layout."
        exit 1
    fi
}

echo "corpus"
link "$DATA/images"                                  data/images
link "$DATA/annotations/annotations_qc_passed.csv"   data/processed/annotations_qc_passed.csv
link "$DATA/annotations/annotations_raw.csv"         data/annotations_pseudonymized.csv
link "$DATA/annotations/human_signals.csv"           data/processed/human_signals.csv
link "$MODELS/cv_hacm/cv_master_full.csv"            data/processed/cv_master_full.csv
link "$DATA/splits/cv_pooled.csv"                    data/processed/cv_pooled.csv
for s in train val test calibration; do
    link "$DATA/splits/$s.csv"                       "data/$s.csv"
done

echo "per-fold tables and fitted temperatures"
link "$MODELS/cv_hacm"                               data/processed/cv_hacm

echo "checkpoints"
link "$MODELS/ensemble_folds"                        checkpoints/cv_module_a
link "$MODELS/baseline_folds"                        checkpoints/cv_baselines
link "$MODELS/ablation"                              checkpoints/ablation
link "$MODELS/mu_h_regressor.pt"                     checkpoints/mu_h_regressor.pt

echo
echo "done. run scripts from $HERE, for example:"
echo "    python analysis/cv_hacm.py"
echo
echo "ImageNet-16H is not part of these deposits. The transfer analyses of"
echo "Section 7.5 expect it at data/imagenet16h/; it is released by Steyvers"
echo "et al. at https://osf.io/2ntrf/"
