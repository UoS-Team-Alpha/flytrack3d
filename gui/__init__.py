"""Interactive tools.

    annotation_tool  start / end point annotation across the views of a session
                     (main_tracks3d_guided.py)
"""

from .annotation_tool import ViewInput, run_annotation_gui, view_input_from_source

__all__ = ["ViewInput", "run_annotation_gui", "view_input_from_source"]
