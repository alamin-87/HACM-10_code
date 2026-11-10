import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HACM10Dataset(Dataset):
    """
    Loads images + human signals. Supervision target is true_label_idx
    (from filename), NOT the annotator majority vote.
    """

    def __init__(self, csv_path, images_dir, transform, class_names):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.class_names = class_names

        missing = []
        for fname in self.df['filename']:
            prefix = fname.split('_')[0]
            path = os.path.join(images_dir, prefix, fname)
            if not os.path.exists(path):
                missing.append(path)

        if missing:
            print(f"WARNING: {len(missing)} image files not found. First few:")
            for p in missing[:5]:
                print(" ", p)
        else:
            print(f"All {len(self.df)} image paths verified OK.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']
        prefix = fname.split('_')[0]
        path = os.path.join(self.images_dir, prefix, fname)

        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        return {
            'image': image,
            'label': int(row['true_label_idx']),
            'filename': fname,
            'mu_H': float(row['mu_H']),
            'sigma_H': float(row['sigma_H']),
            'H_label': float(row['H_label']),
            'degradation_count': int(row['consensus_degradation_count']),
        }