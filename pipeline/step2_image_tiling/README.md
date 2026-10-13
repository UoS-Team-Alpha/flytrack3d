# Step 2: image tiling

Part of the [pipeline](../README.md), which explains how to run the scripts.

## `main_tile_dataset.py`

Tiles a full-frame YOLO dataset into small square crops (`crop_size`), to train a detector that is then run with sliced (SAHI) inference. Its settings are the `tile` section of `configs/image_tiling.json`.

- **Object tiles:** each image gives tiles at random positions around its objects, so that every object ends up in a tile. A box cut by a tile's edge is kept only if enough of it is visible.
- **Background tiles:** a fraction of the images (`empty_fraction`, plus every image without labels) also gives an empty background tile, and these are spread evenly over a 3x3 grid of the frame.
- **Output:** a YOLO dataset with a `data.yaml`, ready for training ([step 4](../step4_training/README.md)).
- **Check:** afterwards it shows a few random tiles with their boxes drawn, to check the labels.
- **Flags:**
  - `--no-visualise` skips the check.
  - `--visualise-only` views tiles made earlier.
  - `--fraction 0.1` tiles 10% of each split as a quick test.
