import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


from vae import VAE
from utils import save_generated_images


def train(model, dataloader, optimizer, epoch, device):
    model.train()
    train_loss = 0.0
    for i, (data, _) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}"
    ):
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mean, log_var = model(data)
        reconstruction_loss = F.binary_cross_entropy(
            reconstruction, data, reduction="sum"
        )  # reconstruction loss, binary cross entropy is used because the input data is binary
        kl_divergence_loss = -0.5 * torch.sum(
            1 + log_var - mean.pow(2) - log_var.exp()
        )  # KL divergence loss
        loss = reconstruction_loss + kl_divergence_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {train_loss / len(dataloader.dataset)}")


if __name__ == "__main__":
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 28 * 28  # For MNIST dataset, input dimension is 28x28 = 784
    hidden_dim = 400
    latent_dim = 20

    batch_size = 128
    epochs = 10
    learning_rate = 1e-3

    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(
        datasets.MNIST("../mnist-data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in tqdm(range(epochs)):
        train(model, train_loader, optimizer, epoch, device)
        save_generated_images(model, train_loader, epoch, device)
