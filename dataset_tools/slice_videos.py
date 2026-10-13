"""Cut the same time window out of every video in a folder, with ffmpeg.

Each video is cut with a stream copy (no re-encoding), so it is fast and
lossless, and every stream of the video is kept. A stream copy can only start
on a keyframe, so a slice may begin slightly before the requested start.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def slice_video(video: Path, output: Path, start: str, end: str) -> None:
    """Write the start..end window of a video (times as HH:MM:SS or seconds) to output."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-ss", str(start), "-to", str(end), "-i", str(video),
                    "-c", "copy", "-map", "0", str(output)], check=True)


def slice_videos(videos_dir: str | Path, start: str, end: str, output_dir: str | Path | None = None,
                 pattern: str = "*.mp4", overwrite: bool = False) -> Path:
    """Slice every video in videos_dir matching pattern. Returns the folder of slices.

    The slices keep the videos' file names. output_dir defaults to
    <videos_dir>/sliced. Slices that already exist are skipped unless overwrite.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found on the PATH. Install it to slice videos.")
    videos_dir = Path(videos_dir)
    if not videos_dir.is_dir():
        raise SystemExit(f"Video folder not found: {videos_dir}")
    videos = sorted(v for v in videos_dir.glob(pattern) if v.is_file())
    if not videos:
        raise SystemExit(f"No videos matching '{pattern}' in {videos_dir}")
    output_dir = Path(output_dir) if output_dir else videos_dir / "sliced"
    if output_dir.resolve() == videos_dir.resolve():
        raise SystemExit("output_dir is the video folder: the slices would overwrite the videos.")
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, video in enumerate(videos, 1):
        output = output_dir / video.name
        if output.exists() and not overwrite:
            print(f"[{i}/{len(videos)}] {video.name}: slice already exists, skipped")
            continue
        print(f"[{i}/{len(videos)}] {video.name}: {start} -> {end}")
        try:
            slice_video(video, output, start, end)
        except subprocess.CalledProcessError:
            output.unlink(missing_ok=True)   # do not leave a broken slice behind
            failed.append(video.name)
    if failed:
        raise SystemExit(f"ffmpeg failed on {len(failed)} of {len(videos)} videos: {', '.join(failed)}")
    return output_dir
