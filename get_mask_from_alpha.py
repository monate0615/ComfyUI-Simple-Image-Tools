from PIL import Image
import torch
import numpy as np
from torch import Tensor

def tensor2pil(image: Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image.Image) -> Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class GetMaskFromAlpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", )
    RETURN_NAMES = ("core_image", "mask_image",)
    FUNCTION = "get_mask"

    CATEGORY = "image/"

    def get_mask(self, image):
        image = tensor2pil(image)

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Separate the RGBA channels
        _, _, _, alpha = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2], image_array[:, :, 3]

        # Define an alpha threshold (e.g., 128 for 50% transparency)
        alpha_threshold = 128

        # Create a mask based on the alpha value (True where alpha > threshold)
        mask = alpha > alpha_threshold

        # Create an empty array to store the segmented image (with transparent background)
        segmented_image_array = np.zeros_like(image_array)

        # Apply the mask: copy RGB values where the mask is True
        segmented_image_array[mask] = image_array[mask]

        # Convert the segmented array back to a PIL image
        segmented_image = pil2tensor(Image.fromarray(segmented_image_array).convert("RGB"))
        mask = pil2tensor(Image.fromarray(mask).convert("RGB"))
        return (segmented_image, mask, )
