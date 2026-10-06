"""Central configuration: paths, model hubs, dataset registry, defaults."""

from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent

ROOT = Path(os.environ.get("GUIDE_ROOT", PKG_ROOT / "runs")).resolve()

DATA_DIR = Path(os.environ.get("GUIDE_DATA", PKG_ROOT / "data")).resolve()

PROBE_DIR = Path(os.environ.get("GUIDE_PROBES", ROOT / "probes")).resolve()

WEIGHTSTATS_DIR = Path(os.environ.get("GUIDE_WEIGHTSTATS", ROOT / "weightstats")).resolve()

SCORE_DIR = ROOT / "scores"
POST_DIR = ROOT / "postprocessed"
EVAL_DIR = ROOT / "evaluation"


def ensure_dirs() -> None:
    for d in (DATA_DIR, PROBE_DIR, SCORE_DIR, POST_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)


TARGET_DATASETS = [
    "aircraft",
    "caltech101",
    "cars",
    "cifar10",
    "cifar100",
    "dtd",
    "flowers",
    "food",
    "pets",
    "sun397",
    "voc2007",
]

DATASET_ALIASES = {
    "sun": "sun397",
    "voc": "voc2007",
    "flowers102": "flowers",
    "food101": "food",
    "oxford_pets": "pets",
    "oxfordpets": "pets",
    "stanford_cars": "cars",
    "fgvc_aircraft": "aircraft",
    "mini-imagenet": "mini_imagenet",
    "miniimagenet": "mini_imagenet",
}

NUM_CLASSES = {
    "aircraft": 100,
    "caltech101": 102,
    "cars": 196,
    "cifar10": 10,
    "cifar100": 100,
    "dtd": 47,
    "flowers": 102,
    "food": 101,
    "pets": 37,
    "sun397": 397,
    "voc2007": 20,
    "mini_imagenet": 100,
    "imagenet": 1000,
}

SOURCE_DATA = "imagenet"
SOURCE_PROBE_DATASET = os.environ.get("GUIDE_SOURCE", "imagenet")

SOURCE_PROBE_SAMPLES = int(os.environ.get("GUIDE_SOURCE_SAMPLES", "5000"))


def canonical_dataset(name: str) -> str:
    n = name.strip().lower()
    return DATASET_ALIASES.get(n, n)


from guide import ground_truth as _gt

CNN_HUB = _gt.HUBS["sfda_cnn"]
CNN_SSL_HUB = _gt.HUBS["sfda_cnn_ssl"]
VIT_HUB = _gt.HUBS["sfda_vit"]
ITM_HUB = _gt.HUBS["itm"]

VIT_SUP_HUB = ["vit_t_16", "vit_s_16", "vit_b_16", "pvt_t", "pvt_s", "pvt_m", "pvtv2_b2", "swin_t"]
VIT_SSL_HUB = [
    "mae_vitb16", "mae_vitl16", "dino_vitb16", "dino_vits8", "dino_vits16",
    "mocov3_vitb16", "mocov3_vits16", "simmim_vitb16",
]

HUBS = {
    "cnn": CNN_HUB,
    "sfda_cnn": CNN_HUB,
    "cnn_ssl": CNN_SSL_HUB,
    "sfda_cnn_ssl": CNN_SSL_HUB,
    "vit": VIT_HUB,
    "sfda_vit": VIT_HUB,
    "vit_sup": VIT_SUP_HUB,
    "vit_ssl": VIT_SSL_HUB,
    "itm": ITM_HUB,
    "all": _gt.ALL_GT_MODELS,
}

DEFAULT_HUB = "sfda_cnn"
DEFAULT_GROUND_TRUTH = _gt.DEFAULT_GT


def resolve_hub(name_or_list) -> list[str]:
    """`"cnn"` -> the CNN hub; `"resnet50,vit_b_16"` -> the two models."""
    if isinstance(name_or_list, (list, tuple)):
        return list(name_or_list)
    key = str(name_or_list).strip().lower()
    if key in HUBS:
        return list(HUBS[key])
    return [m.strip() for m in key.split(",") if m.strip()]


SEED = 42
BATCH_SIZE = 64
IMAGE_SIZE = 224

WEIGHT_VERSION = os.environ.get("GUIDE_WEIGHTS", "v1")

KEEP_AUX_HEADS = os.environ.get("GUIDE_KEEP_AUX", "1").lower() not in ("0", "false", "no")

SAMPLES_PER_CLASS: str | int = "all"


MAX_FEATURE_DIM = None

RESIZE_MODE = os.environ.get("GUIDE_RESIZE", "center_crop")

DISABLED_SCORES: set[str] = {
    "nleep",
    "pactran",
}
if "GUIDE_DISABLE" in os.environ:
    DISABLED_SCORES = {s.strip() for s in os.environ["GUIDE_DISABLE"].split(",") if s.strip()}
