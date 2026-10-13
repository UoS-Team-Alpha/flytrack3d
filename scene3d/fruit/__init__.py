"""The red fake fruit, located without annotations.

    hsv/segment.py       red HSV mask and round candidate blobs in one view
    mesh/reconstruct.py  the candidate combination that triangulates, radius and axis
    mesh/draw.py         the fruit in 3D axes, orthographic panels and camera images
    locate.py            per-session glue and the <run>/3d/fruit/ cache
"""

from .locate import fruit_settings, locate_fruit
from .mesh.draw import draw_fruit_3d, draw_fruit_image, draw_fruit_panel, fruit_points
from .mesh.reconstruct import Fruit

__all__ = ["Fruit", "draw_fruit_3d", "draw_fruit_image", "draw_fruit_panel", "fruit_points",
           "fruit_settings", "locate_fruit"]
