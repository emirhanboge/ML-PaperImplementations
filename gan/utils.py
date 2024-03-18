from torchvision import transforms


def save_image(tensor, filename):
    # [-1, 1] => [0, 1]
    tensor = tensor.view(1, 28, 28)  # Reshape to 28x28
    tensor = (tensor + 1) / 2  # Normalize to [0, 1]
    tensor = tensor.clamp(0, 1)  # Clamp to [0, 1]
    grid = transforms.ToPILImage()(tensor)  # Convert to PIL image
    grid.save(filename)  # Save the image
