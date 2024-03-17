import torch
import torch.nn.functional as F


def compute_content_loss(target_features, content_features):
    """
    Compute the content loss
    """
    target_content = target_features["conv4_2"]  # Get the features of the target image
    content = content_features["conv4_2"]  # Get the features of the content image
    content_loss = F.mse_loss(
        target_content, content
    )  # Compute the mean squared error loss
    return content_loss


def compute_style_loss(target_features, style_features, style_weights):
    """
    Compute the style loss.
    """
    style_loss = 0
    for layer, weight in zip(style_features.keys(), style_weights):
        target_feature = target_features[layer]
        style_feature = style_features[layer]
        
        # Compute the gram matrix of the target image, correlation between the different feature maps
        target_gram = torch.einsum('bchw,bdhw->bcd', target_feature, target_feature)
        
        # Compute the gram matrix of the style image, correlation between the different feature maps
        style_gram = torch.einsum('bchw,bdhw->bcd', style_feature, style_feature)
        
        # Compute the mean squared error loss, normalized by the number of elements in the feature maps
        layer_style_loss = F.mse_loss(target_gram, style_gram, reduction='mean') / (target_feature.shape[-1] * target_feature.shape[-2])
        style_loss += weight * layer_style_loss
    return style_loss


def compute_total_variation_loss(image):
    """
    Compute the total variation loss
    This is used to smooth the image
    """
    # Compute the total variation loss by summing the absolute differences between the pixels
    tv_loss = torch.sum(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])) + \
              torch.sum(torch.abs(image[:, 1:, :, :] - image[:, :-1, :, :]))
    return tv_loss
