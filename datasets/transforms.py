from torchvision import transforms

# ImageNet normalization stats -- used because our backbones (ResNet, EfficientNet,
# MobileNet, ViT) are all pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transforms():
    """
    Single transform pipeline used for ALL splits (train/val/test/calibration).

    No augmentation: augmented views would have no corresponding human
    annotation (mu_H, confidence), which would make HACM comparisons invalid.

    No cropping: direct resize to (224, 224) instead of resize+center-crop,
    so the model sees exactly the same full image content the human
    annotators judged -- important especially for the Occlusion condition,
    where cropping could cut off the already-partially-hidden object.
    """
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])