# Datasets — where to get them

No data is stored in this repository; the working tree it was packaged from is around 550 GB. This
file tells you what to download and where to put it so the scripts find it.

Set `CARLA2REAL_DATA` to wherever you keep the bulk data (default `./datasets`).

## Training corpus

Every source below is public and downloadable. The videos of unestablished provenance that earlier
models used are **no longer part of either corpus**.

### Sunny — 41,646 pairs (`training_pz`)

| Share | Dataset | Download | Licence |
|---:|---|---|---|
| 19,293 | **Mapillary Vistas** (training + validation) | https://www.mapillary.com/dataset/vistas | Free for research. Registration and acceptance of their terms required. |
| 10,000 | **Zenseact Open Dataset** — Frames, DNAT images | https://zod.zenseact.com · devkit https://github.com/zenseact/zod | **CC BY-SA 4.0** (devkit MIT). Commercial use permitted, **share-alike**. |
| 8,240 | **PandaSet** — front camera | https://huggingface.co/datasets/georghess/pandaset · https://pandaset.org | **CC BY 4.0** plus Scale AI / Hesai Dataset Terms. Commercial use permitted, no share-alike. |
| 4,113 | **Cityscapes** (`leftImg8bit` + `gtFine`) | https://www.cityscapes-dataset.com/downloads/ | Free for research. Registration required. |

### Night — 12,546 pairs (`training_pz_night`)

| Share | Dataset | Download | Licence |
|---:|---|---|---|
| 5,000 | **Zenseact Open Dataset** — Frames, DNAT, measured night | as above | **CC BY-SA 4.0** |
| 4,320 | **PandaSet** — 18 night scenes, three forward cameras | as above | **CC BY 4.0** |
| 2,670 | **Dark Zurich** | https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/ | **Check before commercial use** — released for academic research. |
| 556 | Mapillary Vistas night subset, and 534 frames from the withdrawn video set | see note below | mixed |

### Getting PandaSet

One archive, ~44.5 GB, ungated. `LICENSE.txt` sits in every scene directory — read it, because the
Dataset Terms add conditions on top of CC BY 4.0 (no use of the Scale AI or Hesai name or logo
beyond attribution; no use of the data to identify anyone; derived data carries the same terms).

### Getting ZOD

Request access at `opendataset@zenseact.com`; they reply with a personal Dropbox link. The devkit
CLI is the smooth path, and the data is also mirrored on Academic Torrents:

```bash
pip install zod
zod download --url "<your dropbox link>" --output-dir <dir> \
    --subset frames --version full --dnat --no-blur \
    --no-lidar --no-oxts --no-annotations --no-infos --rm -y
```

**Use `--dnat`, not the default `--blur`.** Both are anonymised, but blur leaves smeared patches in
the image while DNAT paints synthetic faces and plates over them. A generator trained on blurred
regions learns to *paint blur*, which is the artefact this pipeline exists to remove.

Images only is ~48 GB of archives; lidar, oxts and annotations are most of the full download and
nothing here reads them.

### Two things to check before you rely on this

**Dark Zurich is still in the night corpus** (2,670 pairs, 21%) and is released for academic
research. If the night model needs to be commercially clean, that share has to be replaced — ZOD
night alone is large enough to do it.

**Mapillary Vistas and Cityscapes are "free for research".** Together they are 56% of the sunny
corpus. That is fine for research use and is a question worth answering before commercial use.
PandaSet and ZOD are the only two sources here that permit commercial use outright.

## Simulator

**CARLA 0.9.16** — https://github.com/carla-simulator/carla/releases — needed only to record new
drives. MIT licence for the code; assets are licensed separately.

## Models used to build the conditioning channels

Not redistributed here. Install from upstream:

| Purpose | Project |
|---|---|
| Depth and surface normals | MoGe — https://github.com/microsoft/MoGe |
| Semantic labels from video | Mask2Former — https://github.com/facebookresearch/Mask2Former |
| Optional temporal stage | Deep Video Prior — https://github.com/ChenyangLEI/deep-video-prior |
| Optional upscaling | Real-ESRGAN — https://github.com/xinntao/Real-ESRGAN |

## Expected layout

```
$CARLA2REAL_DATA/
├── mapillary_vistas/
│   ├── training/{images,v2.0/labels}/
│   └── validation/{images,v2.0/labels}/
├── training_v11_city/           Cityscapes, converted to the Mapillary-65 label space
│   ├── train_img/  train_label/
├── training_pz/                 sunny corpus (Mapillary + ZOD + PandaSet + Cityscapes)
│   ├── train_img/  train_label/  train_edge/  train_depth/  train_normal/  train_chroma/  train_texture/
├── training_pz_night/           night corpus (ZOD + PandaSet + Dark Zurich + Mapillary night)
│   ├── train_img/  train_label/  train_edge/  train_depth/  train_normal/  train_light/
├── recorded_<Town>_<weather>_inst/
│   ├── rgb/  semantic/
└── training_v12_mapillary/      per-town inference channels
    └── test_<Town>_<weather>_inst_gt_{label,edge,depth,normal,chroma,light}/
```

## Label space

Mapillary Vistas 65 classes (`--label_nc 65`). Ids referenced throughout the code: 6 wall, 13 road,
15 sidewalk, 17 building, 24 lane marking, 27 sky, 30 vegetation, 45/46 pole, 50 fence, 55 car,
61 truck/bus. Cityscapes and CARLA labels are both remapped into this space.

## A trap worth knowing

CARLA writes BGRA. Slicing `[:, :, :3]` gives you BGR-as-RGB, so the channels must be reversed
before use. Getting it wrong is silent — the image still looks plausible, just wrong — and it will
poison every downstream channel.

## The 21 training videos

The remaining 28% of the corpus is a set of 21 driving videos whose provenance is unresolved — they
are not redistributable and are not linked here. `TRAINING_VIDEOS.md` in this directory documents
what they are, why they are withheld, and gives a checksum manifest so a copy can be identified.
It also explains what training without them costs (in short: nothing for the sunny baselines).
