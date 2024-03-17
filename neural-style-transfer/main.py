import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

import utils
import model
import losses


def main():
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    content_img = utils.load_image("images/xxx.png", device)
    style_img = utils.load_image("images/xxxbr.png", device)

    vgg = model.get_vgg_model().to(device)

    target = content_img.clone().requires_grad_(True).to(device)

    alpha = 1e0  # Content loss weight
    betas = [1e3, 1e2, 1e1, 1, 1e-1]  # Style loss weights
    tv_weight = 1e-2

    optimizer = torch.optim.Adam(
        [target], lr=0.004
    )  # Target is the image we want to optimize
    steps = 3000
    for step in tqdm(range(steps)):
        target_features = model.get_features(
            target, vgg
        )  # Get the features of the target image
        content_features = model.get_features(
            content_img, vgg
        )  # Get the features of the content image
        style_features = model.get_features(
            style_img, vgg
        )  # Get the features of the style image

        content_loss = losses.compute_content_loss(
            target_features, content_features
        )  # Compute the content loss
        style_loss = losses.compute_style_loss(
            target_features, style_features, betas
        )  # Compute the style loss
        tv_loss = losses.compute_total_variation_loss(target)

        total_loss = alpha * content_loss + style_loss + tv_weight * tv_loss

        optimizer.zero_grad()
        total_loss.backward()  # Compute the gradients of the total loss w.r.t the target image
        optimizer.step()  # Update the target image

        if step % 100 == 0 or step == steps - 1:
            print(f"Step [{step}/{steps}], Total Loss: {total_loss.item():.4f}")
            utils.save_image(target, f"images/xxx/image_{step}.png")


if __name__ == "__main__":
    main()
