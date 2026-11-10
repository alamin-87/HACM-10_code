import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
from datasets.ground_truth import get_true_label, CLASS_TO_IDX


def compute_human_signals(df):
    """
    Aggregates per-annotator rows into one row per image.
    true_label comes from the filename (real ground truth), while the
    annotators' majority label is kept separately as a perception signal.
    """
    records = []

    for filename, group in df.groupby('filename'):
        confidences = group['confidence'].values / 100.0
        labels = group['labelName'].values
        n_annotators = len(group)

        true_label = get_true_label(filename)
        majority_label = pd.Series(labels).mode().iloc[0]

        mu_H = confidences.mean()
        sigma_H = confidences.std() if n_annotators > 1 else 0.0

        label_counts = pd.Series(labels).value_counts()
        label_probs = label_counts / label_counts.sum()
        H_label = scipy_entropy(label_probs, base=2)

        conditions = group['condition_canonical'].values
        condition_counts = pd.Series(conditions).value_counts()
        consensus_condition = condition_counts.index[0]
        condition_agreement = condition_counts.iloc[0] / n_annotators

        degradation_counts = group['degradation_count'].values
        consensus_degradation_count = pd.Series(degradation_counts).mode().iloc[0]

        human_correct_rate = (labels == true_label).mean()

        records.append({
            'filename': filename,
            'true_label': true_label,
            'true_label_idx': CLASS_TO_IDX[true_label],
            'majority_label': majority_label,
            'majority_correct': majority_label == true_label,
            'human_correct_rate': human_correct_rate,
            'mu_H': mu_H,
            'sigma_H': sigma_H,
            'H_label': H_label,
            'n_annotators': n_annotators,
            'consensus_condition': consensus_condition,
            'condition_agreement': condition_agreement,
            'consensus_degradation_count': consensus_degradation_count,
        })

    result_df = pd.DataFrame(records)

    print(f"Images processed: {len(result_df)}")
    print(f"Images with all 3 annotators: {(result_df['n_annotators'] == 3).sum()}")
    print(f"Majority-vote accuracy vs true label: {result_df['majority_correct'].mean()*100:.1f}%")
    print(f"Mean per-annotator accuracy: {result_df['human_correct_rate'].mean()*100:.1f}%")

    return result_df