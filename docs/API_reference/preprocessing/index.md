---
weight: 10
title: Preprocessing
---

# Preprocessing

Turning raster and image data into batches of chips that a detector can consume. This is where chip
size, stride, resolution, coordinate reference system, and region of interest are decided.

- [Preprocessing Docstrings](preprocessing.md) — the `create_dataloader`,
  `create_intersection_dataloader`, and `create_image_dataloader` functions, plus visualization and
  saving of dataloader contents.

- [Derived Geodatasets Docstrings](derived_geodatasets.md) — the `torchgeo` dataset and data module
  subclasses that back those dataloaders.

- [Utils Docstrings](utils.md) — image transforms applied during loading, such as Gaussian and box
  blur.
