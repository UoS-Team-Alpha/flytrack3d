"""Cut the same time window out of every video in a folder.

The folder of videos, the window (start / end) and the output folder come from
the "slice" section of a video config. Each video is cut with an ffmpeg stream
copy (fast, no re-encoding) and written under the same file name to the output
folder (<videos_dir>/sliced by default). The videos themselves are not changed.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
    python pipeline/step0_video_editing/main_slice_videos.py --start 02:07:00 --end 02:08:00
    python pipeline/step0_video_editing/main_slice_videos.py --videos path/to/videos --output path/to/slices --overwrite
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from dataset_tools.video_tools.slice_videos import slice_videos  # noqa: E402
from common.config import load_config  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "video_config.json"   # "videos_dir" and "slice" hold the folder and the window
VIDEOS_DIR = None          # folder of videos to slice; None = the config's
START = None               # start of the window, e.g. "02:07:00" or seconds; None = the config's
END = None                 # end of the window, e.g. "02:08:00"; None = the config's
OUTPUT_DIR = None          # folder for the slices; None = the config's
OVERWRITE = None           # True = redo slices that already exist; None = the config's
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Cut the same time window out of every video in a folder.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--videos", default=VIDEOS_DIR, help="Folder of videos to slice.")
    parser.add_argument("--start", default=START, help="Start of the window, HH:MM:SS or seconds.")
    parser.add_argument("--end", default=END, help="End of the window, HH:MM:SS or seconds.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Folder for the slices.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE,
                        help="Redo slices that already exist.")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config not found: {args.config}\n"
                         f"Copy config_examples/video_config.example.json to video_config.json and fill it in.")
    config = load_config(args.config)
    if "slice" not in config:
        raise SystemExit("config has no 'slice' section. See config_examples/video_config.example.json.")
    s = config["slice"]
    overrides = {"start": args.start, "end": args.end, "output_dir": args.output, "overwrite": args.overwrite}
    s = {**s, **{k: v for k, v in overrides.items() if v is not None}}

    output_dir = slice_videos(args.videos or config.get("videos_dir") or "", s["start"], s["end"],
                              output_dir=s.get("output_dir"), pattern=s.get("pattern", "*.mp4"),
                              overwrite=bool(s.get("overwrite")))
    print(f"\nSlices saved to: {output_dir}")


if __name__ == "__main__":
    main()
