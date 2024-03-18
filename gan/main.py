import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from train import train
from models import Generator, Discriminator

if __name__ == "__main__":
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    dataset = datasets.MNIST(
        root="../mnist-data/", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    G = Generator(latent_size=64, hidden_size=256, image_size=28 * 28).to(device)
    D = Discriminator(image_size=28 * 28, hidden_size=256).to(device)

    train(G, D, dataloader, device, epochs=200, latent_size=64)
