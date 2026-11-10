import torch.nn as nn
import timm


def build_efficientnet_b4(num_classes=10, pretrained=True, drop_rate=0.3):
    """
    EfficientNet-B4 backbone for Module A's ensemble. Same drop_rate=0.3
    for MC Dropout compatibility.
    """
    model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    return model