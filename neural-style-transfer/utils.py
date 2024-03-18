import torch
from torchvision import transforms

from PIL import Image


def load_image(image_path, device):
    max_size = 400
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    transform = transforms.Compose(
        [
            transforms.Resize(size),  # Resize the image to the desired size
            transforms.ToTensor(),  # Convert the image to a tensor with values between 0 and 1
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),  # Normalizing the image
        ]
    )

    image = transform(image)[:3, :, :].unsqueeze(
        0
    )  # This is to add the batch dimension

    return image.to(device)


def save_image(tensor, filename):
    # Clone the tensor to not do changes in-place
    image = tensor.cpu().clone().squeeze(0)

    # Denormalize the image
    denormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    image = denormalize(image)

    # Clip the values to be between 0 and 1 (image data format)
    image = torch.clamp(image, 0, 1)

    # Convert to a PIL image and save
    image = transforms.ToPILImage()(image)
    image.save(filename)
