# What is not covered by the MIT license

**English** · [Deutsch](NOTICE.de.md)

The [MIT license](LICENSE) applies to the **code** in this repository.
It does not cover:

## Mapillary data

`data/<city>/processed/metadata.parquet` **is in the repository** (about
66 MB across six cities) and contains Mapillary metadata: `image_id`,
`sequence_id`, `captured_at`, `lat`, `lon`, `compass_angle`, `is_pano`,
`creator_id`. Likewise `cache/detections.jsonl` (image ID and count per
semantic class).

These data are licensed under [CC BY-SA 4.0](https://www.mapillary.com/terms).
Attribution: **Images and metadata from [Mapillary](https://www.mapillary.com),
CC BY-SA 4.0.** Anyone who publishes datasets derived from them must pass them
on under the same terms.

The **image collection** is not in the repository — `notebooks/03_image_download`
downloads it with your own API token to `image_root/<city>`.

An exception are the figures under `results/<city>/figures/demo/`:
they show individual query and database images side by side to make matches
and misses visible, and thus contain Mapillary image content.
As a derivative work they are under the same terms — CC BY-SA 4.0,
**images from Mapillary, CC BY-SA 4.0**. The same applies to these
figures when they appear in talks or reports.

CC BY-SA also requires the **author of every single image**, not
only the platform. Under section 3(a)(2) of the license a link
to a page that names the author is sufficient — the image's page on Mapillary.
[`QUELLEN.md`](results/osnabrueck/figures/demo/QUELLEN.md) in the same folder
links every image of every figure; `src/quellen.py` writes the file
whenever the demo or `experiments/beispielbilder.py` saves a
figure. Anyone reusing a figure takes the links along.

`creator_id` is a pseudonymous Mapillary account ID. Together with `lat`,
`lon` and `captured_at` it yields a capture trail per account. It is
not in the metadata for decoration: the "Hard" ground truth in
`src/evaluation.py` needs exactly this comparison ("different photographer or
more than 180 days apart"), and without it the duplicate effect
in the city comparison could not be measured. Mapillary automatically blurs
faces and licence plates before publication; this applies to the images,
not to the metadata.

## OpenStreetMap

City boundary, road network and districts come from OpenStreetMap via
OSMnx/Overpass. These data are licensed under the
[ODbL](https://www.openstreetmap.org/copyright). Attribution:
**© OpenStreetMap contributors**. This concerns all maps under
`results/<city>/figures/` and `experiments/results/<city>/*.png` as well as
the district columns in the corresponding JSONs.

## Third-party repositories and model weights

None of this is in the repository. `setup_external.py`, `torch.hub`,
Hugging Face and `pip` fetch it onto your own machine on the first run;
the license of each project applies, not the MIT license here.
Checked on 2026-09-23 against the projects' license files, and for the
Hugging Face weights against the model card.

| Component | Source | License | Copyright |
|---|---|---|---|
| **MegaLoc** — code | `torch.hub`, [gmberton/MegaLoc](https://github.com/gmberton/MegaLoc) | MIT | © 2024 Gabriele Berton, Carlo Masone |
| **MegaLoc** — weights | Hugging Face, [gberton/MegaLoc](https://huggingface.co/gberton/MegaLoc) (`model.safetensors`) | MIT (according to the model card) | the same |
| **EigenPlaces** — code and weights | `torch.hub`, [gmberton/EigenPlaces](https://github.com/gmberton/EigenPlaces), loads parts of [gmberton/CosPlace](https://github.com/gmberton/CosPlace) | MIT | © 2023 Berton, Trivigno, Masone, Caputo; CosPlace © 2022 Berton, Masone, Caputo |
| **AnyLoc** — code and vocabulary | `external/AnyLoc` ([AnyLoc/AnyLoc](https://github.com/AnyLoc/AnyLoc)), vocabulary from the project's Hugging Face Space | BSD-3-Clause | © 2023 AnyLoc |
| **DINOv2 ViT-G/14** — weights for AnyLoc | `torch.hub`, [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0 | Meta Platforms |
| **MixVPR** — code and weights | `external/MixVPR` ([amaralibey/MixVPR](https://github.com/amaralibey/MixVPR)), weights from Google Drive | **no license file** — all rights reserved by the authors | Ali-bey, Chaib-draa, Giguère |
| **CLIP ViT-B/32** — weights | Hugging Face, `openai/clip-vit-base-patch32`, via 🤗 Transformers (Apache-2.0) | MIT ([openai/CLIP](https://github.com/openai/CLIP)) | © 2021 OpenAI |
| **LightGlue** — code and weights | `pip`, [cvg/LightGlue](https://github.com/cvg/LightGlue) | Apache-2.0 | Lindenberger, Sarlin, Pollefeys |
| **SuperPoint** — weights and inference code | installed with LightGlue | **[Magic Leap, non-commercial research only](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE)** | Magic Leap, Inc. |

Two entries restrict what you may do with the project:

- **SuperPoint** may only be used for your own non-commercial research
  and may not be redistributed. This affects only
  `experiments/geometric_verification.py`; the pipeline 01–08, `locate.py`
  and the demo work without SuperPoint. Anyone who wants to use the geometric
  verification commercially needs a different detector —
  LightGlue is also available for DISK (Apache-2.0) and ALIKED (BSD-3-Clause).
- **MixVPR** has no license file. Without a license nothing is explicitly
  permitted; the project clones code and weights only onto your own machine
  to re-measure the published numbers, and redistributes none of it.

None of the components requires a citation. The authors ask for one; the
entries are in the README under "Literature".
