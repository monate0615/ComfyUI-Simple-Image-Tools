from PIL import Image
import torch
import numpy as np
from torch import Tensor

def tensor2pil(image: Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image.Image) -> Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tan(x):
    if x == 18:
        return 1e+9
    else:
        return np.tan(x * np.pi / 36.)
    
def cot(x):
    if x == 0:
        return 1e+9
    else:
        return 1 / np.tan(x * np.pi / 36.)

class GetQuadrilateralOutfit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", )
    RETURN_NAMES = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",)
    FUNCTION = "get_quadrilateral_outfit"

    CATEGORY = "image/"

    def get_quadrilateral_outfit(self, image):
        image = tensor2pil(image)

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Get height and width
        height, width = image_array[:2]

        # Set length step
        l_step = min(height, width) / 100

        # Initialize angles
        angle_1, angle_2, angle_3, angle_4 = 1
        x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = 0

        # Process
        tmp = width * height
        flag = False
        while angle_1 <= 18:
            y1 = 0
            while y1 < height:
                x = 0
                while x < min(width, y1 * tan(angle_1)):
                    if image_array[int(y1 - x * cot(angle_1)), x, 0] > 128:
                        flag = True
                        break
                    else:
                        x = x + 1
                if flag:
                    break
                else:
                    y1 = y1 + l_step
            flag = False
            
            while angle_2 <= 18:
                x2 = width - 1
                while x2 >= 0:
                    y = 0
                    while y < min(height, (width - x2) * tan(angle_2)):
                        if image_array[y, int(x2 + y * cot(angle_2)), 0] > 128:
                            flag = True
                            break
                        else:
                            y = y + 1
                    if flag:
                        break
                    else:
                        x2 = x2 - l_step
                flag = False

                while angle_3 <= 18:
                    y3 = height - 1
                    while y3 >= 0:
                        x = width - 1
                        while x >= max(0, width - (height - y3) * tan(angle_3)):
                            if image_array[int(y3 + (width - x) * cot(angle_3)), x, 0] > 128:
                                flag = True
                                break
                            else:
                                x = x - 1
                        if flag:
                            break
                        else:
                            y3 = y3 - l_step
                    flag = False

                    while angle_4 <= 18:
                        x4 = 0
                        while x4 < width:
                            y = height - 1
                            while y >= max(0, width - x4 * tan(angle_4)):
                                if image_array[y, int(x4 - (height - y) * cot(angle_4)), 0] > 128:
                                    flag = True
                                    break
                                else:
                                    y = y - 1
                            if flag:
                                break
                            else:
                                x4 = x4 + l_step
                        flag = False

                        # Calculate the black part of quadrilateral
                        black = 0
                        for i in range(width):
                            for j in range(height):
                                if j > y1 - i * cot(angle_1) and \
                                    i < x2 + j * cot(angle_2) and \
                                    j < y3 + (width - i) * cot(angle_3) and \
                                    i > x4 - (height - j) * cot(angle_4) and \
                                    image_array[j, i, 0] < 128:
                                    black = black + 1

                        if black < tmp:
                            tmp = black
                            
                            x_1 = (y1 + x2 * tan(angle_2)) / (tan(angle_2) + cot(angle_1))
                            y_1 = (y1 * cot(angle_1) - x2) / (tan(angle_1) + cot(angle_2))
                            x_2 = (x2 + y3 * tan(angle_3)) / (tan(angle_3) + cot(angle_2))
                            y_2 = (x2 * cot(angle_2) - y3) / (tan(angle_2) + cot(angle_3))
                            x_3 = (y3 + x4 * tan(angle_4)) / (tan(angle_4) + cot(angle_3))
                            y_3 = (y3 * cot(angle_3) - x4) / (tan(angle_3) + cot(angle_4))
                            x_4 = (x4 + y1 * tan(angle_1)) / (tan(angle_1) + cot(angle_4))
                            y_4 = (x4 * cot(angle_4) - y1) / (tan(angle_4) + cot(angle_1))

        return (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, )
