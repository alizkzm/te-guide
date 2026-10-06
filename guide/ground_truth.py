"""Ground-truth fine-tuning accuracies."""

from __future__ import annotations

import csv
import json
from pathlib import Path

DATASETS = [
    "aircraft", "caltech101", "cars", "cifar10", "cifar100", "dtd",
    "flowers", "food", "pets", "sun397", "voc2007",
]


def _table(rows: dict[str, list[float]], datasets: list[str] = DATASETS) -> dict:
    """rows = {model: [acc per dataset]} -> {dataset: {model: acc}}."""
    out: dict[str, dict[str, float]] = {d: {} for d in datasets}
    for model, accs in rows.items():
        if len(accs) != len(datasets):
            raise ValueError(f"{model}: {len(accs)} accuracies for {len(datasets)} datasets")
        for d, a in zip(datasets, accs):
            out[d][model] = float(a)
    return out


SFDA_CNN = _table({
    "resnet34":     [84.06, 91.15, 88.63, 96.12, 81.94, 72.96, 95.20, 81.99, 93.50, 61.02, 84.60],
    "resnet50":     [84.64, 91.98, 89.09, 96.28, 82.80, 74.72, 96.26, 84.45, 93.88, 63.54, 85.80],
    "resnet101":    [85.53, 92.38, 89.47, 97.39, 84.88, 74.80, 96.53, 85.58, 93.92, 63.76, 85.68],
    "resnet152":    [86.29, 93.10, 89.88, 97.53, 85.66, 76.44, 96.86, 86.28, 94.42, 64.82, 86.32],
    "densenet121":  [84.66, 91.50, 89.34, 96.45, 82.75, 74.18, 97.02, 84.99, 93.07, 63.26, 85.28],
    "densenet169":  [84.19, 92.51, 89.02, 96.77, 84.26, 74.72, 97.32, 85.84, 93.62, 64.10, 85.77],
    "densenet201":  [85.38, 93.14, 89.44, 97.02, 84.88, 76.04, 97.10, 86.71, 94.03, 64.57, 85.67],
    "mnasnet1_0":   [66.48, 89.34, 72.58, 92.59, 72.04, 70.12, 95.39, 71.35, 91.08, 56.56, 81.06],
    "mobilenet_v2": [79.68, 88.64, 86.44, 94.74, 78.11, 71.72, 96.20, 81.12, 91.28, 60.29, 82.80],
    "googlenet":    [80.32, 90.85, 87.76, 95.54, 79.84, 72.53, 95.76, 79.30, 91.38, 59.89, 82.58],
    "inception_v3": [80.15, 92.75, 87.74, 96.18, 81.49, 72.85, 95.73, 81.76, 92.14, 59.98, 83.84],
})

SFDA_CNN_HEAD = _table({
    "resnet34":     [38.19, 89.80, 32.04, 78.61, 59.43, 66.70, 90.71, 60.56, 91.27, 71.96, 82.46],
    "resnet50":     [40.63, 89.75, 50.91, 83.57, 65.41, 70.74, 93.05, 65.79, 91.76, 83.29, 83.28],
    "resnet101":    [41.21, 89.81, 50.60, 85.24, 67.64, 69.57, 92.30, 66.50, 92.34, 75.61, 83.85],
    "resnet152":    [42.98, 91.42, 52.07, 85.33, 67.81, 70.74, 93.06, 67.55, 92.67, 75.72, 84.13],
    "densenet121":  [43.61, 90.03, 51.78, 81.39, 62.11, 68.09, 93.23, 65.37, 91.46, 76.37, 82.73],
    "densenet169":  [47.15, 90.76, 56.20, 83.08, 64.53, 69.95, 94.15, 67.81, 92.60, 80.78, 84.07],
    "densenet201":  [46.39, 91.31, 57.32, 84.52, 67.51, 70.64, 93.01, 68.11, 92.57, 80.38, 83.34],
    "mnasnet1_0":   [41.72, 87.85, 46.19, 69.55, 37.49, 65.69, 92.37, 62.65, 89.56, 79.63, 81.18],
    "mobilenet_v2": [42.24, 87.35, 49.77, 76.97, 57.46, 67.77, 92.27, 62.60, 89.73, 73.25, 80.88],
    "googlenet":    [36.22, 88.31, 43.83, 78.45, 59.73, 66.12, 89.53, 55.34, 89.41, 76.82, 80.32],
    "inception_v3": [28.21, 88.48, 27.60, 69.87, 46.39, 61.28, 83.01, 46.31, 85.85, 63.72, 77.01],
})

SFDA_CNN_SSL = _table({
    "byol":          [82.10, 91.90, 89.83, 96.98, 83.86, 76.37, 96.80, 85.44, 91.48, 63.69, 85.13],
    "deepclusterv2": [82.43, 91.16, 90.16, 97.17, 84.84, 77.31, 97.05, 87.24, 90.89, 66.54, 85.38],
    "infomin":       [83.78, 80.86, 86.90, 96.72, 70.89, 73.47, 95.81, 78.82, 90.92, 57.67, 81.41],
    "insdis":        [79.70, 77.21, 80.21, 93.08, 69.08, 66.40, 93.63, 76.47, 84.58, 51.62, 76.33],
    "mocov1":        [81.85, 79.68, 82.19, 94.15, 71.23, 67.36, 94.32, 77.21, 85.26, 53.83, 77.94],
    "mocov2":        [83.70, 82.76, 85.55, 96.48, 71.27, 72.56, 95.12, 77.15, 89.06, 56.28, 78.32],
    "pclv1":         [82.16, 88.60, 87.15, 96.42, 79.44, 73.28, 95.62, 77.70, 88.93, 58.36, 81.91],
    "pclv2":         [83.00, 87.52, 85.56, 96.55, 79.84, 69.30, 95.87, 80.29, 88.72, 58.82, 81.85],
    "selav2":        [85.42, 90.53, 89.85, 96.85, 84.36, 76.03, 96.22, 86.37, 89.61, 65.74, 85.52],
    "simclrv1":      [80.54, 90.94, 89.98, 97.09, 84.49, 73.97, 95.33, 82.20, 88.53, 63.46, 83.29],
    "simclrv2":      [81.50, 88.58, 88.82, 96.22, 78.91, 74.71, 95.39, 82.23, 89.18, 60.93, 83.08],
    "swav":          [83.04, 89.49, 89.81, 96.81, 83.78, 76.68, 97.11, 87.22, 90.59, 66.10, 85.06],
})

SFDA_VIT = _table({
    "vit_t_16":      [71.26, 89.39, 82.09, 96.52, 81.58, 71.86, 95.50, 81.96, 91.44, 58.40, 83.10],
    "vit_s_16":      [73.12, 92.70, 86.72, 97.69, 86.62, 75.08, 96.79, 86.26, 94.02, 64.76, 86.62],
    "vit_b_16":      [78.39, 93.47, 89.26, 98.56, 89.96, 77.66, 97.98, 88.96, 94.61, 68.62, 87.88],
    "pvtv2_b2":      [84.14, 93.13, 90.60, 97.96, 88.24, 77.16, 97.89, 88.67, 93.86, 66.44, 86.44],
    "pvt_t":         [69.76, 90.04, 84.10, 94.87, 75.26, 72.92, 95.80, 83.78, 91.48, 61.86, 84.60],
    "pvt_s":         [75.20, 93.02, 87.61, 97.34, 86.20, 75.77, 97.32, 86.98, 94.13, 65.78, 86.62],
    "pvt_m":         [76.70, 93.75, 87.66, 97.93, 87.36, 77.10, 97.36, 85.56, 94.48, 67.22, 87.36],
    "swin_t":        [81.90, 91.90, 88.93, 97.34, 85.97, 77.04, 97.40, 86.67, 94.50, 65.51, 87.54],
    "mocov3_vits16": [76.04, 89.84, 82.18, 97.92, 85.84, 71.88, 93.89, 82.84, 90.44, 60.60, 81.84],
    "dino_vits16":   [72.18, 86.76, 79.81, 97.96, 85.66, 75.96, 95.96, 85.69, 92.59, 64.14, 84.80],
})

ITM = _table({
    "densenet121":   [83.86, 97.23, 88.39, 97.38, 85.67, 69.47, 90.41, 85.00, 91.77, 72.49, 82.75],
    "densenet161":   [88.24, 98.21, 89.84, 97.79, 87.01, 72.61, 92.00, 86.72, 93.24, 74.10, 82.27],
    "densenet169":   [83.80, 97.29, 88.83, 97.75, 86.24, 70.90, 91.58, 85.58, 92.91, 73.76, 81.85],
    "densenet201":   [84.04, 97.47, 89.55, 97.65, 86.39, 72.98, 91.07, 86.14, 93.08, 74.13, 80.86],
    "efficientnet_b0": [81.61, 97.87, 87.25, 97.88, 87.01, 68.88, 89.71, 85.82, 90.68, 73.87, 81.10],
    "googlenet":     [83.29, 96.54, 86.54, 96.97, 83.39, 69.73, 88.01, 81.36, 90.60, 69.67, 80.45],
    "inception_v3":  [79.60, 96.95, 84.73, 96.92, 83.42, 66.75, 87.14, 81.79, 92.86, 68.07, 79.50],
    "mnasnet1_0":    [75.01, 95.56, 81.12, 96.78, 83.40, 60.96, 76.48, 82.87, 90.65, 70.02, 78.78],
    "mobilenet_v2":  [76.36, 96.03, 85.65, 96.48, 82.15, 67.34, 90.03, 83.69, 89.02, 71.16, 82.45],
    "resnet18":      [77.56, 95.74, 83.58, 96.57, 82.69, 65.59, 88.52, 79.92, 88.50, 68.71, 76.74],
    "resnet34":      [77.53, 96.54, 86.11, 97.30, 85.31, 67.50, 90.03, 85.24, 92.70, 70.75, 81.56],
    "resnet50":      [84.49, 97.38, 88.87, 97.70, 84.60, 70.85, 91.38, 87.08, 93.65, 74.85, 84.19],
    "resnet101":     [85.75, 97.47, 88.71, 98.14, 87.58, 71.17, 90.91, 87.97, 94.00, 77.21, 83.83],
    "resnet152":     [81.64, 97.64, 89.07, 98.01, 88.58, 71.44, 89.25, 88.35, 94.79, 76.18, 84.73],
    "dino_vitb16":   [78.40, 98.16, 88.42, 98.70, 90.53, 74.31, 93.19, 87.88, 92.97, 77.02, 84.51],
    "dino_vits8":    [80.02, 97.47, 89.33, 98.65, 90.24, 71.97, 90.84, 90.84, 93.10, 77.53, 85.58],
    "mocov3_vitb16": [75.70, 97.70, 88.53, 98.68, 90.74, 72.07, 93.61, 87.15, 89.75, 75.92, 80.37],
    "mae_vitb16":    [72.16, 97.52, 88.16, 98.42, 87.55, 69.04, 85.80, 87.30, 89.81, 75.29, 82.86],
    "mae_vitl16":    [85.30, 97.98, 91.21, 98.55, 91.28, 74.26, 90.73, 90.82, 94.69, 79.18, 88.01],
    "simmim_vitb16": [68.32, 96.20, 86.02, 98.63, 88.80, 66.17, 83.87, 87.88, 87.16, 74.55, 79.62],
})


TABLES: dict[str, dict] = {
    "sfda": SFDA_CNN,
    "sfda_cnn": SFDA_CNN,
    "sfda_cnn_full": SFDA_CNN,
    "sfda_cnn_head": SFDA_CNN_HEAD,
    "sfda_cnn_ssl": SFDA_CNN_SSL,
    "sfda_vit": SFDA_VIT,
    "itm": ITM,
    "cnn": SFDA_CNN,
    "vit": SFDA_VIT,
}

HUBS: dict[str, list[str]] = {name: sorted({m for d in t.values() for m in d})
                              for name, t in TABLES.items()}

ALL_GT_MODELS: list[str] = sorted(
    {m for t in (SFDA_CNN, SFDA_CNN_SSL, SFDA_VIT, ITM) for d in t.values() for m in d}
)

DEFAULT_GT = "sfda_cnn"


def get_ground_truth(name: str = DEFAULT_GT) -> dict[str, dict[str, float]]:
    key = str(name).strip().lower()
    if key in TABLES:
        return TABLES[key]
    return load_ground_truth(name)


def load_ground_truth(path: str | Path) -> dict[str, dict[str, float]]:
    """JSON `{dataset: {model: acc}}` or CSV with dataset,model,accuracy columns."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ground truth {path!r} is neither a built-in table "
            f"({', '.join(sorted(TABLES))}) nor an existing file"
        )
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))

    out: dict[str, dict[str, float]] = {}
    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cols = {k.strip().lower(): v for k, v in row.items()}
            d = cols.get("dataset") or cols.get("target_data")
            m = cols.get("model") or cols.get("model_name")
            a = cols.get("accuracy") or cols.get("acc") or cols.get("value")
            if d and m and a not in (None, ""):
                out.setdefault(d, {})[m] = float(a)
    return out


def hub_for(ground_truth: dict[str, dict[str, float]] | str) -> list[str]:
    """Every model that appears anywhere in a table (or a named table)."""
    if isinstance(ground_truth, str):
        ground_truth = get_ground_truth(ground_truth)
    models: set[str] = set()
    for per_dataset in ground_truth.values():
        models.update(per_dataset)
    return sorted(models)


def datasets_for(ground_truth: dict[str, dict[str, float]] | str) -> list[str]:
    if isinstance(ground_truth, str):
        ground_truth = get_ground_truth(ground_truth)
    return [d for d in DATASETS if ground_truth.get(d)]
