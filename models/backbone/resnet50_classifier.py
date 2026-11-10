import torch.nn as nn
import timm


def build_resnet50_classifier(num_classes=10, pretrained=True, drop_rate=0.3):
    """
    Single ResNet-50 classifier used for the loss ablation.

    Kept deliberately simple: the only thing that varies across ablation
    configs is the loss function, so any difference in results is
    attributable to the loss and not to architecture or capacity.
    Returns raw logits; losses apply softmax themselves where needed.
    """
    model = timm.create_model(
        'resnet50',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model
