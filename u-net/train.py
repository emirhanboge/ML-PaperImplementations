import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from u_net import UNet
from utils import mask_transform, overlay_mask


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=1).to(device).float()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    target_transform = transforms.Compose(
        [
            transforms.Resize(
                (64, 64), interpolation=transforms.InterpolationMode.NEAREST
            ),
            mask_transform,
        ]
    )
    dataset = datasets.VOCSegmentation(
        root="../vosegmentation-data",
        year="2012",
        image_set="train",
        download=True,
        transform=transform,
        target_transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i, (data, target) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        ):
            data = data.to(device).float()
            target = target.to(device).float()

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                for i, (data, target) in enumerate(dataloader):
                    if i == 0:  # Save only the first 5 images
                        data = data.to(device).float()
                        output = model(data)
                        output = torch.sigmoid(output).cpu().numpy()
                        for j in range(len(data)):
                            if j < 5:  # Save only the first 5 images
                                original_image = transforms.ToPILImage()(data[j].cpu())
                                segmented_image = (output[j][0] * 255).astype(np.uint8)
                                overlayed_image = overlay_mask(
                                    original_image, segmented_image, color=(255, 0, 0)
                                )  # Overlay the mask in red
                                os.makedirs("segmented_images", exist_ok=True)
                                original_image.save(
                                    f"segmented_images/original_{epoch}_{i * len(data) + j}.png"
                                )
                                overlayed_image.save(
                                    f"segmented_images/segmented_{epoch}_{i * len(data) + j}.png"
                                )

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    print("Training finished.")
