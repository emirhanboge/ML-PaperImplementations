import os
from PIL import Image

import torch
import torchvision.utils as vutils
import numpy as np


def save_generated_images(model, dataloader, epoch, device, num_images=10):
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        data, _ = next(iter(dataloader))
        data = data.to(device)
        data = data.view(data.size(0), -1)
        reconstruction, _, _ = model(data)

        reconstruction = reconstruction[:num_images]
        reconstruction = reconstruction.view(reconstruction.size(0), 1, 28, 28)
        reconstruction_images = reconstruction.cpu()

        grid = vutils.make_grid(reconstruction_images, nrow=num_images, pad_value=1)

        np_grid = grid.numpy().transpose((1, 2, 0))
        np_grid = (np_grid * 255).astype(np.uint8)

        pil_img = Image.fromarray(np_grid)
        generated_images_dir = "generated_images"
        os.makedirs(generated_images_dir, exist_ok=True)
        file_path = f"{generated_images_dir}/epoch_{epoch}.png"
        pil_img.save(file_path)

        print(f"Generated images with labels saved to {file_path}")
