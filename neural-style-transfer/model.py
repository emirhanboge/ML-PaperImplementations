import torch
from torchvision import models


def get_vgg_model():
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

    # Repalce MaxPool with AvgPool
    for i, layer in enumerate(vgg):
        if isinstance(layer, torch.nn.MaxPool2d):
            vgg[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    # Freeze all the layers in order to keep the weights fixed
    for param in vgg.parameters():
        param.requires_grad_(False)

    return vgg


def get_features(image, model):
    layers = {
        "0": "conv1_1",  # Style layers
        "5": "conv2_1",  # Style layers
        "10": "conv3_1",  # Style layers
        "19": "conv4_1",  # Style layers
        "21": "conv4_2",  # Content layer (for content representation)
        "28": "conv5_1",  # Style layers
    }

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
        if name == "28":
            break

    return features
