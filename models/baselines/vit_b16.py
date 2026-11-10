import torch.nn as nn
import timm


def build_vit_b16(num_classes=10, pretrained=True):
    """
    Loads a ViT-B/16 (Vision Transformer) backbone pretrained on ImageNet,
    with its final classification layer replaced for our 10 classes.

    NOTE: ViT is heavier than ResNet-18/MobileNetV2 -- expect this to be
    noticeably slower per epoch on CPU, and it needs 224x224 input, which
    our transforms.py already provides.
    """
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    return model