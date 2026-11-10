import torch
import torch.nn as nn

from models.backbone.resnet50 import build_resnet50
from models.backbone.efficientnet_b4 import build_efficientnet_b4


class EnsembleBackbone(nn.Module):
    """
    Module A's core: ResNet-50 + EfficientNet-B4, combined via equal-weight
    softmax averaging (Section 5.1). This produces p_AI(x), the AI confidence
    signal used everywhere downstream (HACM, CMP loss, etc.).
    """

    def __init__(self, num_classes=10, pretrained=True, drop_rate=0.3):
        super().__init__()
        self.resnet = build_resnet50(num_classes=num_classes, pretrained=pretrained, drop_rate=drop_rate)
        self.effnet = build_efficientnet_b4(num_classes=num_classes, pretrained=pretrained, drop_rate=drop_rate)

    def forward(self, x):
        logits_resnet = self.resnet(x)
        logits_effnet = self.effnet(x)

        p_resnet = torch.softmax(logits_resnet, dim=1)
        p_effnet = torch.softmax(logits_effnet, dim=1)

        p_AI = 0.5 * p_resnet + 0.5 * p_effnet  # equal-weight ensemble (Section 5.1)
        return p_AI