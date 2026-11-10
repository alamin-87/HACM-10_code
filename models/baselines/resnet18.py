import torch.nn as nn
import timm


def build_resnet18(num_classes=10, pretrained=True):
    """
    Loads a ResNet-18 backbone pretrained on ImageNet, with its final
    classification layer replaced to output our 10 classes instead of
    ImageNet's 1000.
    """
    model = timm.create_model('resnet18', pretrained=pretrained, num_classes=num_classes)
    return model