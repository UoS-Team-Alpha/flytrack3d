"""Static arena geometry drawn into the 3D plots of tracking3d, found without annotations.

    fruit    the red fake fruit: HSV segmentation per view (fruit/hsv), triangulation
             and its mesh (fruit/mesh), cached per run by fruit/locate.py
    leaves   the green tape star around it: HSV outline per view (leaves/hsv), laid onto
             the fruit's wall and voted across views (leaves/mesh)
    arena    the frame of pipes (a tapered box), grown out from the fruit by its measured
             size (arena/mesh)
    objects  ArenaObjects: every object of a session, located and drawn together

Shared by the objects:
    geometry  the fruit's wall, its up / right axes, camera rays meeting it
    drawing   style merging, edge colours, reprojection into camera images
    frames    the median frame of a view
    settings  config["arena_plotting"] defaults and the per-run cache
"""

from .objects import ArenaObjects, locate_arena

__all__ = ["ArenaObjects", "locate_arena"]
