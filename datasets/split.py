import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def make_stratify_key(df, min_count=8):
    """
    Stratification key from true_label x consensus_degradation_count.
    Rare combinations are merged into a class-level fallback so sklearn
    can still split them four ways.
    """
    df = df.copy()
    key = df['true_label'].astype(str) + '_' + df['consensus_degradation_count'].astype(str)
    counts = key.value_counts()
    rare_keys = counts[counts < min_count].index

    key_final = key.where(~key.isin(rare_keys), df['true_label'].astype(str) + '_rare')
    df['stratify_key'] = key_final

    n_rare_rows = key.isin(rare_keys).sum()
    print(f"Stratify key: {key_final.nunique()} groups ({len(rare_keys)} rare combos merged, affecting {n_rare_rows} rows)")
    return df


def stratified_split(df, test_size=0.15, calibration_size=0.15, val_size=0.15, seed=42):
    """
    Splits into train / val / test / calibration, stratified by
    (true_label, degradation_count).
    """
    df = make_stratify_key(df, min_count=8)

    train_val_cal, test = train_test_split(
        df, test_size=test_size, stratify=df['stratify_key'], random_state=seed
    )

    remaining_fraction_for_cal = calibration_size / (1 - test_size)
    train_val, calibration = train_test_split(
        train_val_cal, test_size=remaining_fraction_for_cal,
        stratify=train_val_cal['stratify_key'], random_state=seed
    )

    remaining_fraction_for_val = val_size / (1 - test_size - calibration_size)
    train, val = train_test_split(
        train_val, test_size=remaining_fraction_for_val,
        stratify=train_val['stratify_key'], random_state=seed
    )

    for name, split_df in [('train', train), ('val', val), ('test', test), ('calibration', calibration)]:
        print(f"{name}: {len(split_df)} images ({len(split_df) / len(df) * 100:.1f}%)")

    return (
        train.drop(columns=['stratify_key']),
        val.drop(columns=['stratify_key']),
        test.drop(columns=['stratify_key']),
        calibration.drop(columns=['stratify_key']),
    )