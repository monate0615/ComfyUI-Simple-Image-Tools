from .get_mask_from_alpha import GetMaskFromAlpha
from .get_quadrilateral_outfit import GetQuadrilateralOutfit

NODE_CLASS_MAPPINGS = {
    "GetMaskFromAlpha": GetMaskFromAlpha,
    "GetQuadrilateralOutfit": GetQuadrilateralOutfit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetMaskFromAlpha": "Get Mask From Alpha",
    "GetQuadrilateralOutfit": "Get Quadrilateral Outfit",
}