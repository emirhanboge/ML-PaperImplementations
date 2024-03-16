import os

import torch
import torchvision.datasets as datasets
from torchvision import transforms

from alexnet import AlexNet

if __name__ == "__main__":
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize the image to 256x256
            transforms.ToTensor(),  # Convert the image to a tensor with pixels in the range [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize the image with the ImageNet specific values
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=os.path.join(os.path.dirname(__file__), "data"),
        train=True,
        download=True,
        transform=data_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=os.path.join(os.path.dirname(__file__), "data"),
        train=False,
        download=True,
        transform=data_transform,
    )

    # For demonstration purposes, we will use a subset of the dataset
    train_dataset.data = train_dataset.data[:1000]
    train_dataset.targets = train_dataset.targets[:1000]
    test_dataset.data = test_dataset.data[:100]
    test_dataset.targets = test_dataset.targets[:100]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AlexNet(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} done")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f"Accuracy after epoch {epoch}: {100 * correct / total}%")

    print("Training done")
