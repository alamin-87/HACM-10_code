import torch.nn as nn
import timm


def build_mobilenetv2(num_classes=10, pretrained=True):
    """
    Loads a MobileNetV2 backbone pretrained on ImageNet, with its final
    classification layer replaced to output our 10 classes.
    """
    model = timm.create_model('mobilenetv2_100', pretrained=pretrained, num_classes=num_classes)
    return model