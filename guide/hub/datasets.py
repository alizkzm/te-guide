"""The 11 target datasets + the mini-ImageNet source probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets as tvds
from torchvision import transforms as T

from guide import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def build_transform(
    image_size: int = 224, normalise: bool = True, resize_mode: str | None = None
) -> T.Compose:
    """`resize_mode` in {"squash", "center_crop"}; default `config.RESIZE_MODE`."""
    mode = str(resize_mode or config.RESIZE_MODE).lower()
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if normalise else ([0.0] * 3, [1.0] * 3)

    if mode == "squash":
        resize = [T.Resize((image_size, image_size))]
    elif mode in ("center_crop", "centercrop", "crop"):
        resize = [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
        ]
    else:
        raise ValueError(f"unknown resize_mode {mode!r}; use 'squash' or 'center_crop'")

    return T.Compose(
        resize
        + [
            T.Lambda(lambda im: im.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


class VOC2007Classification(Dataset):
    """Multi-label VOC2007 collapsed to a single label (argmax of the multi-hot)."""

    def __init__(self, root, image_set="trainval", transform=None, download=False):
        self.base = tvds.VOCDetection(
            root=str(root), year="2007", image_set=image_set,
            transform=None, download=download,
        )
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, target = self.base[i]
        objs = target["annotation"]["object"]
        if isinstance(objs, dict):
            objs = [objs]
        multi_hot = np.zeros(len(VOC_CLASSES), dtype=np.float32)
        for o in objs:
            name = o["name"]
            if name in VOC_CLASSES:
                multi_hot[VOC_CLASSES.index(name)] = 1.0
        label = int(multi_hot.argmax())
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class HFImageDataset(Dataset):
    """Thin torch wrapper over a HuggingFace image dataset."""

    def __init__(self, hf_ds, transform, image_key="image", label_key="label"):
        self.ds, self.transform = hf_ds, transform
        self.image_key, self.label_key = image_key, label_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[int(i)]
        img = row[self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        return self.transform(img), int(row[self.label_key])


def _load_mini_imagenet(transform, split="train"):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "mini-ImageNet needs the HuggingFace `datasets` package: "
            "pip install datasets"
        ) from exc
    hf_split = "train" if split in ("train", "trainval", "all") else "validation"
    return HFImageDataset(load_dataset("timm/mini-imagenet", split=hf_split), transform)


def _load_imagenet(root: Path, split: str, transform):
    """ImageNet-1k - the real source domain of every model in the hub."""
    import os

    split = "train" if split == "train" else "val"
    candidates = []
    env = os.environ.get("GUIDE_IMAGENET_DIR")
    if env:
        candidates += [Path(env) / split, Path(env)]
    candidates += [root / "imagenet" / split, root / "ImageNet" / split, root / split]

    try:
        return tvds.ImageNet(str(root / "imagenet"), split=split, transform=transform)
    except Exception:
        pass

    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return tvds.ImageFolder(str(path), transform=transform)

    raise FileNotFoundError(
        "ImageNet-1k is the source probe but cannot be downloaded automatically.\n"
        f"    Put the '{split}' images at {root / 'imagenet' / split}/<wnid>/*.JPEG\n"
        f"    (or set GUIDE_IMAGENET_DIR), then re-run.\n"
        "    Alternatives: GUIDE_SOURCE=mini_imagenet uses the 100-class HuggingFace\n"
        "    stand-in instead, and only `otce` strictly needs a source probe at all."
    )


def _load_cars(root, split, transform):
    """torchvision's StanfordCars download is dead; fall back to HuggingFace."""
    try:
        return tvds.StanfordCars(root=str(root), split=split, transform=transform, download=True)
    except Exception:
        from datasets import load_dataset

        hf_split = "train" if split == "train" else "test"
        return HFImageDataset(load_dataset("tanganke/stanford_cars", split=hf_split), transform)


def _load_sun397(root: Path, transform):
    """SUN397 - Princeton's tarball URL 404s, so torchvision's download fails."""
    if (Path(root) / "SUN397").is_dir():
        return tvds.SUN397(str(root), transform=transform, download=False)
    try:
        return tvds.SUN397(str(root), transform=transform, download=True)
    except Exception as exc:
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                f"SUN397 download failed ({exc}) and the HuggingFace fallback needs "
                f"`pip install datasets`.\n"
                f"    Or extract SUN397.tar.gz yourself to {Path(root) / 'SUN397'}."
            ) from exc
        parts = []
        for hf_split in ("train", "test"):
            try:
                parts.append(HFImageDataset(
                    load_dataset("tanganke/sun397", split=hf_split), transform))
            except Exception:
                continue
        if not parts:
            raise RuntimeError(
                f"SUN397 is unavailable: torchvision's download 404s ({exc}) and the "
                f"HuggingFace mirror could not be read either.\n"
                f"    Fix by extracting SUN397.tar.gz to {Path(root) / 'SUN397'},\n"
                f"    or drop the dataset:  -d aircraft,caltech101,cars,cifar10,cifar100,"
                f"dtd,flowers,food,pets,voc2007"
            ) from exc
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def build_dataset(name: str, split: str, transform, root: Path | None = None) -> Dataset:
    """`split` in {"train", "val", "test"}."""
    name = config.canonical_dataset(name)
    root = Path(root or config.DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)

    if name == "cifar10":
        if split == "val":
            raise KeyError("cifar10 has no val split")
        return tvds.CIFAR10(root, train=(split == "train"), transform=transform, download=True)
    if name == "cifar100":
        if split == "val":
            raise KeyError("cifar100 has no val split")
        return tvds.CIFAR100(root, train=(split == "train"), transform=transform, download=True)
    if name == "aircraft":
        s = {"train": "train", "val": "val", "test": "test"}[split]
        return tvds.FGVCAircraft(root, split=s, annotation_level="variant",
                                 transform=transform, download=True)
    if name == "dtd":
        return tvds.DTD(root, split=split, partition=1, transform=transform, download=True)
    if name == "flowers":
        return tvds.Flowers102(root, split=split, transform=transform, download=True)
    if name == "food":
        if split == "val":
            raise KeyError("food101 has no val split")
        return tvds.Food101(root, split=("train" if split == "train" else "test"),
                            transform=transform, download=True)
    if name == "pets":
        if split == "val":
            raise KeyError("oxford pets has no val split")
        return tvds.OxfordIIITPet(root, split=("trainval" if split == "train" else "test"),
                                  transform=transform, download=True)
    if name == "caltech101":
        if split != "train":
            raise KeyError("caltech101 ships a single split")
        return tvds.Caltech101(root, target_type="category", transform=transform, download=True)
    if name == "sun397":
        if split != "train":
            raise KeyError("sun397 ships a single split")
        return _load_sun397(root, transform)
    if name == "cars":
        if split == "val":
            raise KeyError("stanford cars has no val split")
        return _load_cars(root, "train" if split == "train" else "test", transform)
    if name == "voc2007":
        s = {"train": "train", "val": "val", "test": "test"}[split]
        return VOC2007Classification(root, image_set=s, transform=transform, download=True)
    if name == "mini_imagenet":
        if split == "val":
            raise KeyError("use split='train' or 'test' for mini-imagenet")
        return _load_mini_imagenet(transform, split)
    if name == "imagenet":
        if split in ("val", "test"):
            return _load_imagenet(root, "val", transform)
        if split == "train":
            return _load_imagenet(root, "train", transform)
        raise KeyError(f"imagenet has no {split!r} split")

    raise ValueError(f"unknown dataset {name!r}. known: {', '.join(config.TARGET_DATASETS)}")


def get_dataset(name: str, split: str = "all", image_size: int = 224,
                normalise: bool = True, root: Path | None = None,
                resize_mode: str | None = None) -> Dataset:
    """`split` in {"train", "val", "test", "trainval", "all"}."""
    transform = build_transform(image_size, normalise, resize_mode)
    if split in ("train", "val", "test"):
        return build_dataset(name, split, transform, root)

    wanted = ["train", "val"] if split == "trainval" else ["train", "val", "test"]
    parts = []
    for s in wanted:
        try:
            parts.append(build_dataset(name, s, transform, root))
        except KeyError:
            continue
    if not parts:
        raise RuntimeError(f"no usable split for {name} (requested {split})")
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def train_row_count(dataset: Dataset, split: str) -> int:
    """How many leading rows of `dataset` come from the **train** split."""
    if split in ("val", "test"):
        return 0
    if isinstance(dataset, Subset):
        base = train_row_count(dataset.dataset, split)
        return int(sum(1 for i in dataset.indices if i < base))
    if isinstance(dataset, ConcatDataset):
        return int(dataset.cumulative_sizes[0])
    return len(dataset)


def subsample(dataset: Dataset, n: int | None, seed: int = 1234) -> Dataset:
    if n is None or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return Subset(dataset, sorted(idx))


def make_loader(dataset: Dataset, batch_size: int = 64, num_workers: int = 0,
                pin_memory: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
