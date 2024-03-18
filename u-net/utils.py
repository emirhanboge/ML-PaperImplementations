from PIL import Image
import numpy as np
import torch


def overlay_mask(image, mask, color=(255, 0, 0)):
    """
    Overlays a mask on an image.

    Args:
        image: PIL Image, the original image.
        mask: numpy array, the segmentation mask.
        color: tuple, the color to use for the mask overlay.

    Returns:
        PIL Image, the original image with the mask overlay.
    """
    image = image.convert("RGBA")  # Convert to RGBA for transparency
    overlay = Image.new("RGBA", image.size, color + (0,))  # Create a colored overlay
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            if mask[y, x] > 0:  # If the mask is positive at this pixel
                overlay.putpixel(
                    (x, y), color + (180,)
                )  # Set the overlay pixel to the specified color with transparency
    return Image.alpha_composite(image, overlay).convert(
        "RGB"
    )  # Composite the overlay onto the image and convert back to RGB


def mask_transform(mask):
    mask = np.array(mask)
    mask = mask.astype(np.float32)
    mask[mask == 255] = 0  # Background
    return torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
