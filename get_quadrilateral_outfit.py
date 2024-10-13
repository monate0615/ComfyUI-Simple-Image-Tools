import numpy as np
import cv2
import sympy
from typing import List, Tuple
import torch
from torch import Tensor

def tensor2cv2(image: Tensor) -> np.ndarray:
    img = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0)) 
    return img

def cv22tensor(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    if image.ndim == 3 and image.shape[0] == 3:
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    else:
        tensor = torch.from_numpy(image).unsqueeze(0)
    return tensor

# Function to approximate the best fitting N-gon (quadrilateral)
def appx_best_fit_ngon(mask_cv2: np.ndarray, n: int = 4) -> List[Tuple[int, int]]:
    # Convert the image to grayscale and find contours
    mask_cv2_gray = cv2.cvtColor(mask_cv2, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(mask_cv2_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    hull = np.array(hull).reshape((len(hull), 2))

    # Convert points to sympy Points
    hull = [sympy.Point(*pt) for pt in hull]

    # Reduce the hull points to an n-gon
    while len(hull) > n:
        best_candidate = None
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # Skip if the sum of the angles is not more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # Find intersection of adjacent edges
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # Calculate the area of the triangle and update best candidate
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            if best_candidate and best_candidate[1] < area:
                continue

            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # Convert the hull points back to Python integers
    return [(int(x), int(y)) for x, y in hull]


class GetQuadrilateralOutfit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", )
    RETURN_NAMES = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",)
    FUNCTION = "get_quadrilateral_outfit"
    CATEGORY = "image/"

    def get_quadrilateral_outfit(self, image):
        # Load the image using OpenCV
        img = tensor2cv2(image)

        # Get dimensions of the original image
        original_height, original_width, _ = img.shape

        # Create a black image 2x the size of the original image
        black_image_width = original_width * 2
        black_image_height = original_height * 2
        black_image = np.zeros((black_image_height, black_image_width, 3), dtype=np.uint8)

        # Calculate the position to center the original image
        paste_x = (black_image_width - original_width) // 2
        paste_y = (black_image_height - original_height) // 2

        # Place the original image in the center of the black image
        black_image[paste_y:paste_y + original_height, paste_x:paste_x + original_width] = img

        # Get the approximate quadrilateral around the object in the black image
        hulls = appx_best_fit_ngon(black_image)

        # Adjust the hull coordinates relative to the center of the original image
        res = (
            hulls[0][0] - original_width // 2, hulls[0][1] - original_height // 2,
            hulls[1][0] - original_width // 2, hulls[1][1] - original_height // 2,
            hulls[2][0] - original_width // 2, hulls[2][1] - original_height // 2,
            hulls[3][0] - original_width // 2, hulls[3][1] - original_height // 2,
        )

        return res
