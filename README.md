# MovingImages

Code for generating "moving image" videos from static image datasets and training image/video classifiers on them (image-to-video augmentation).

<!-- PAPER SECTION: title, authors, venue, abstract, figures (put images in assets/) and citation will be added here -->

## Repository structure

```
movingimages/
  data_generation/   moving_dataset.py, ss_moving.py: turn images into moving videos
  models/            ResNet image/video models, GSF/TSM modules (gateshift.py)
  ops/               basic ops (consensus module)
  utils/             datasets, transforms, helpers, Tiny-ImageNet loader
scripts/             training entry points (image, video, augmented, self-supervised, optuna)
assets/              figures for the README
```

## Installation

```
pip install -r requirements.txt
```

## Usage

Run everything from the repository root, as modules:

```
python -m movingimages.data_generation.moving_dataset --dataset <name>
python -m movingimages.data_generation.ss_moving --dataset <name>
python -m scripts.video_train --dataset <name> --root <path> --lr <lr> --wd <wd> --epochs <n>
python -m scripts.image_train
python -m scripts.augmented_train
python -m scripts.self_supervised_train
python -m scripts.optuna_video_train
python -m scripts.optuna_classification
```

Use `--help` on each module for its options.

## Data

Generated datasets are expected in folders such as `moving_cifar-10/`, `moving_mnist/`, `moving_emnist/`, `moving_fashion-mnist/` with `train_videofolder.txt` / `val_videofolder.txt` lists (see `movingimages/utils/helpers.py`). Some scripts contain hard-coded absolute data paths from the original author's machine (e.g. `scripts/augmented_train.py`, `scripts/image_train.py`) that must be edited.

## License

No license file is provided in the original repository.
