import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import save_image


def train(G, D, dataloader, device, epochs=200, latent_size=64):
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=1e-4)
    D_optimizer = optim.Adam(D.parameters(), lr=1e-4)
    k = 1  # Number of steps to apply to the discriminator

    for epoch in tqdm(range(epochs)):
        G_loss = 0.0
        D_loss = 0.0
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)

            # Train Discriminator
            D_optimizer.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)  # Real labels
            fake_labels = torch.zeros(batch_size, 1).to(device)  # Fake labels

            for _ in range(k):
                z = torch.randn(batch_size, latent_size).to(device)  # Random noise
                fake_imgs = G(z)  # Generate fake images
                real_pred = D(imgs)  # Predict real images
                fake_pred = D(fake_imgs.detach())  # Predict fake images

                real_loss = criterion(
                    real_pred, real_labels
                )  # Calculate loss for real images
                fake_loss = criterion(
                    fake_pred, fake_labels
                )  # Calculate loss for fake images
                loss = real_loss + fake_loss  # Total loss for the discriminator
                loss.backward()
                D_loss += loss.item()
                D_optimizer.step()  # Update the discriminator weights

            # Train Generator
            G_optimizer.zero_grad()
            z = torch.randn(batch_size, latent_size).to(device)  # Random noise
            fake_imgs = G(z)  # Generate fake images
            fake_pred = D(fake_imgs)  # Predict fake images
            loss = criterion(fake_pred, real_labels)  # Calculate loss for the generator
            loss.backward()
            G_loss += loss.item()
            G_optimizer.step()  # Update the generator weights

        print(
            f"Epoch [{epoch+1}/{epochs}] G_loss: {G_loss/i:.4f}, D_loss: {D_loss/i:.4f}"
        )
        save_image(fake_imgs[0].cpu().detach(), f"images/sample_{epoch}.png")
