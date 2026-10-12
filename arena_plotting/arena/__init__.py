"""The arena frame (the tapered box of pipes), grown out from the fruit by its measured size.

    mesh/reconstruct.py  the box: back face on the fruit's wall, front face `depth` out
    mesh/draw.py         its pipes and floor in 3D axes, orthographic panels and camera images
    locate.py            per-session glue
"""

from .locate import frame_settings, locate_frame
from .mesh.draw import draw_frame_3d, draw_frame_image, draw_frame_panel
from .mesh.reconstruct import ArenaFrame

__all__ = ["ArenaFrame", "draw_frame_3d", "draw_frame_image", "draw_frame_panel", "frame_settings",
           "locate_frame"]
