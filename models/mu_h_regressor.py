import torch
import torch.nn as nn
import timm


class MuHRegressor(nn.Module):
    """
    Predicts human confidence mu_H directly from the image.

    This is the decisive test for a difficulty-conditioned calibrator: if a
    network trained explicitly on mu_H cannot recover it from pixels, then
    perceived difficulty is not linearly available in the visual features,
    and no post-hoc scheme conditioned on model-side signals can supply it
    either.

    Sigmoid output, since mu_H lies in [0, 1].
    """

    def __init__(self, pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'resnet50', pretrained=pretrained, num_classes=0, drop_rate=drop_rate
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return torch.sigmoid(self.head(feats)).squeeze(1)
