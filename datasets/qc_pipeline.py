import pandas as pd


def parse_conditions(condition_str):
    if condition_str == 'Baseline':
        return frozenset(), 0
    parts = [p.strip() for p in condition_str.split(',')]
    return frozenset(parts), len(parts)


# Threshold per degradation_count -- capped at 3.0s for 2, 3, and 4
DURATION_THRESHOLDS = {
    0: 0,   # Baseline
    1: 2,   # single degradation
    2: 3.0,   # two combined
    3: 3.0,   # three combined
    4: 3.0,   # all four combined
}


def run_qc(df):
    """
    Applies quality-control filters to raw annotation data (Section 4.4).
    """
    df = df.copy()

    df['condition_set'], df['degradation_count'] = zip(*df['ambiguityCondition'].apply(parse_conditions))
    df['condition_canonical'] = df['condition_set'].apply(lambda s: ', '.join(sorted(s)) if s else 'Baseline')

    # 1. Completion time threshold -- degradation-aware, capped at 3.0s
    min_time = df['degradation_count'].map(DURATION_THRESHOLDS)
    before = len(df)
    df = df[df['durationSeconds'] >= min_time]
    print(f"Completion-time filter (degradation-aware): removed {before - len(df)} rows")

    # 2. Zero confidence variance -- window=35
    def flag_zero_variance(group, window=35):
        conf = group['confidence'].values
        for i in range(len(conf) - window + 1):
            if len(set(conf[i:i + window])) == 1:
                return True
        return False

    zero_var_sessions = df.groupby('sessionId').apply(flag_zero_variance)
    bad_sessions = zero_var_sessions[zero_var_sessions].index.tolist()
    before = len(df)
    df = df[~df['sessionId'].isin(bad_sessions)]
    print(f"Zero-variance filter (window=35): removed {before - len(df)} rows across {len(bad_sessions)} sessions")

    # 3. Label entropy check -- window=30
    def flag_identical_labels(group, window=30):
        labels = group['labelName'].values
        for i in range(len(labels) - window + 1):
            if len(set(labels[i:i + window])) == 1:
                return True
        return False

    if len(df) > 0:
        bad_label_sessions = df.groupby('sessionId').apply(flag_identical_labels)
        bad_label_sessions = bad_label_sessions[bad_label_sessions].index.tolist()
        before = len(df)
        df = df[~df['sessionId'].isin(bad_label_sessions)]
        print(f"Label-entropy filter: removed {before - len(df)} rows across {len(bad_label_sessions)} sessions")

    print(f"\nFinal QC-passed row count: {len(df)}")
    return df