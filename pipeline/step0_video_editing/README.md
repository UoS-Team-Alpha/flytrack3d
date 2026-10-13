# Step 0: video editing

Part of the [pipeline](../README.md), which explains how to run the scripts.

## `main_slice_videos.py`

Cuts the same time window (`start` to `end`) out of every video in a folder, using an ffmpeg stream copy (fast, no re-encoding).

- **Needs:** `ffmpeg` on the PATH.
- **Settings:** the folder and the window come from its own config, `configs/video.json` (template: `configs/video.example.json`).
- **Output:** the slices keep the videos' file names and go to `<videos_dir>/sliced` unless `output_dir` is set. Existing slices are skipped.
