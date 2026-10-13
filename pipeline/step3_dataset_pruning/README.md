# Step 3: dataset pruning

Part of the [pipeline](../README.md), which explains how to run the scripts.

A dataset built from tracked video is full of near-identical frames: a fly that sits still gets the same bounding box size in hundreds of consecutive frames. This step finds those frames and builds a pruned copy of the dataset without them. The original dataset is only ever read.

- **Needs:** a dataset laid out as `images/<split>/` + `labels/<split>/`.
- **Settings:** `configs/dataset_pruning.json`, one section per script.
- **Interactive:** both scripts are interactive. Choose options by number; `q` quits at any prompt and Ctrl+C exits cleanly, also while a plot window is open.

| Script | Config section |
| --- | --- |
| [`main_analyse_redundancy.py`](#main_analyse_redundancypy) | `analyse_redundancy` |
| [`main_guided_prune.py`](#main_guided_prunepy) | `guided_prune` |

## `main_analyse_redundancy.py`

Look before you prune; nothing is changed.

How it works:

1. Pick a split and a video prefix (or all prefixes).
2. The boxes are grouped by their exact width and height (augmentation moves a box but does not resize it). Each size is a **rank**: rank 1 is the most common.
3. For the top `ssim_ranks` ranks, each box is cropped from the greyscale image and compared with [SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure) in frame order. A crop is **kept** if it differs (SSIM < `ssim_threshold`) from every crop kept so far, else it is **redundant**.
4. A rank's `distinct / count` ratio says how varied it is: a still fly might give 12 distinct crops out of 782.
5. Then pick saved results and a rank to see sample frames of it with the boxes drawn.

Options:

- `--no-grids` skips the sample frames; `--grids-only` views results saved earlier.
- `decimals` rounds the sizes before grouping them (3 = within 0.64 px at 640 px), so more boxes fall into the top ranks.

Results are reused while the label files and settings are unchanged.

## `main_guided_prune.py`

Walks a whole split one video prefix at a time and builds one pruned dataset at the end.

### For each prefix

1. It runs the analysis above and shows a grid for each SSIM'd rank.
2. It prints which ranks it would prune (those with `distinct / count` below `distinct_cutoff`) and what that drops.
3. It asks `y` to accept or `n` to keep the prefix as it is.

### What gets dropped

- A frame is dropped only if **every** box in it is redundant, so no fly is ever left unlabelled.
- `keep_fraction` / `min_keep_per_rank` of each rank's redundant frames are kept anyway as a safety margin.
- No video loses more than `max_drop_fraction` of its frames.

### Building the pruned dataset

When every prefix is done, it asks whether to build `output/pruned_datasets/<dataset>_pruned_<ddmmyyyy>/` (`_2`, `_3`, ... if built again the same day):

- images hard-linked,
- labels copied,
- image lists (`train.txt`) filtered,
- the other splits copied unchanged.

The before / after counts per split and per video are printed and saved next to it.

### Resuming

Every answer is saved straight away, so after `q` or Ctrl+C the next run offers to continue where you left off (with the session's own settings) or to start again.

### Stricter second pass

After a full pass, the largest videos may still dominate the split, usually because only a small share of their boxes fell into the analysed ranks. To walk only those videos again, set:

- `prefixes` (full video prefixes, or a part that matches exactly one),
- `decimals: 3`,
- a larger `top_n` / `ssim_ranks`,
- a lower `min_keep_per_rank`.

Each video is planned as if its own earlier run were not there: `y` replaces that run, `n` keeps it, and everything is rebuilt once at the end. The subset has its own session, so the full-split one is not affected.

### Review

`--review` (or `--review-only`) shows the frames a run dropped, each next to the kept frame it matched, zoomed in on the fly and labelled with its SSIM.

### What is kept where

Everything is under `output/pruned_datasets/` (`output_dir` to change; keep it on the dataset's drive, or the images are copied instead of hard-linked).

| Folder | Holds |
| --- | --- |
| `bbox_sizes/` | the rank tables, the per-box SSIM decisions and a fingerprint of the label files and settings, by which out-of-date results are refused |
| `purge/` | each run's manifest (`prune_<run_id>.csv`: every dropped frame with the frame it matched), `purge_log.csv` (every run's status: `planned`, `applied`, `declined`, `discarded`, `replaced`) and the guided sessions |

Things to know:

- A pruned dataset is always the original minus every applied run.
- Deleting the original or an old pruned folder does not affect the images in the other folders, but editing an image in place changes it in every folder that links to it.
- YOLO's `labels/*.cache` files are not copied, so they are rebuilt.
- A relative `path:` in `data.yaml` gets a warning, because Ultralytics resolves it against its datasets folder rather than the yaml's own folder.
