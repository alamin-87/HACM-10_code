import torch.nn as nn
import timm


def build_resnet50(num_classes=10, pretrained=True, drop_rate=0.3):
    """
    ResNet-50 backbone for Module A's ensemble. drop_rate=0.3 enables
    dropout layers needed later for MC Dropout (Section 5.1).
    """
    model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    return model