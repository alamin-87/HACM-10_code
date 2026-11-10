import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class AblationDataset(Dataset):
    """
    Dataset for the loss ablation.

    Extends the standard loader with a stable per-sample index, which the
    Correctness Ranking Loss needs in order to accumulate each sample's
    correctness history across epochs. The index refers to the row position
    in the pooled CSV and stays fixed for the whole run.
    """

    def __init__(self, csv_path, images_dir, transform, class_names):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.class_names = class_names

        missing = []
        for fname in self.df['filename']:
            prefix = fname.split('_')[0]
            if not os.path.exists(os.path.join(images_dir, prefix, fname)):
                missing.append(fname)
        if missing:
            print(f"WARNING: {len(missing)} images not found, e.g. {missing[:3]}")
        else:
            print(f"All {len(self.df)} image paths verified OK.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']
        prefix = fname.split('_')[0]
        image = Image.open(os.path.join(self.images_dir, prefix, fname)).convert('RGB')

        return {
            'image': self.transform(image),
            'label': int(row['true_label_idx']),
            'filename': fname,
            'sample_index': idx,
            'mu_H': float(row['mu_H']),
            'sigma_H': float(row['sigma_H']),
            'H_label': float(row['H_label']),
            'degradation_count': int(row['consensus_degradation_count']),
        }
