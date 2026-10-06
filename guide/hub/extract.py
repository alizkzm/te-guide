"""One forward pass per (model, dataset) -> a cached `ProbeData`."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import torch

from guide import config
from guide.core.probe import ProbeData
from guide.hub import datasets as ds
from guide.hub import models as mh


class _HeadTap:
    """Captures the input and output of the classifier head."""

    def __init__(self, head):
        self.head = head
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self._handles = []

    def __enter__(self):
        def pre_hook(_m, args):
            x = args[0]
            self.inputs.append(x.detach().reshape(x.shape[0], -1).float().cpu())

        def post_hook(_m, _args, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.outputs.append(out.detach().reshape(out.shape[0], -1).float().cpu())

        self._handles = [
            self.head.register_forward_pre_hook(pre_hook),
            self.head.register_forward_hook(post_hook),
        ]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


@torch.no_grad()
def extract_probe(
    model_name: str,
    dataset: str,
    split: str = "all",
    device: str = "cpu",
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,
    max_samples: int | None = None,
    sample_seed: int = 1234,
    source_data: str = config.SOURCE_DATA,
    resize_mode: str | None = None,
    with_flops: bool = False,
    verbose: bool = True,
) -> ProbeData:
    dataset = config.canonical_dataset(dataset)
    t0 = time.perf_counter()

    model = mh.load_model(model_name)
    image_size = mh.image_size_for(model_name)
    head = mh.resolve_classifier(model)
    headless = head is None or mh.is_headless(model_name)

    weights = mh.extract_weight_matrices(model, model_name)
    n_params = mh.count_params(model)
    flops = mh.count_flops(model, image_size) if with_flops else None

    resize_mode = str(resize_mode or config.RESIZE_MODE).lower()
    data = ds.get_dataset(dataset, split=split, image_size=image_size, resize_mode=resize_mode)
    data = ds.subsample(data, max_samples, seed=sample_seed)
    n_train = ds.train_row_count(data, split)
    loader = ds.make_loader(
        data, batch_size=batch_size, num_workers=num_workers, pin_memory=(device != "cpu")
    )
    if verbose:
        print(f"  [{model_name} @ {dataset}] {len(data)} images, head={'no' if headless else 'yes'}")

    model = model.to(device).eval()
    labels: list[torch.Tensor] = []
    feats: list[torch.Tensor] = []

    if headless:
        for x, y in loader:
            out = model(x.to(device, non_blocking=True))
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim == 3:
                out = out[:, 0]
            feats.append(out.detach().reshape(out.shape[0], -1).float().cpu())
            labels.append(y)
        features = torch.cat(feats).numpy().astype(np.float32)
        logits = None
    else:
        with _HeadTap(head) as tap:
            for x, y in loader:
                model(x.to(device, non_blocking=True))
                labels.append(y)
        features = torch.cat(tap.inputs).numpy().astype(np.float32)
        logits = torch.cat(tap.outputs).numpy().astype(np.float32)

    y = torch.cat(labels)
    if y.ndim > 1:
        y = y.argmax(dim=1)
    y = y.numpy().astype(np.int64)

    elapsed = time.perf_counter() - t0
    meta = {
        "model_name": model_name,
        "dataset": dataset,
        "source_data": source_data,
        "split": split,
        "n_train": int(n_train),
        "resize_mode": resize_mode,
        "image_size": image_size,
        "n_params": n_params,
        "flops": flops,
        "feature_dim": int(features.shape[1]),
        "n_source_classes": None if logits is None else int(logits.shape[1]),
        "source_top1": mh.source_top1(model_name),
        "weights_tag": mh.weights_tag(model_name, config.WEIGHT_VERSION),
        "headless": bool(headless),
        "n_samples": int(features.shape[0]),
        "extraction_seconds": elapsed,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weight_shapes": {k: list(v.shape) for k, v in weights.items()},
    }

    probe = ProbeData(
        model_name=model_name,
        dataset=dataset,
        source_data=source_data,
        features=features,
        labels=y,
        logits=logits,
        weights=weights,
        meta=meta,
    )
    probe.save()
    if verbose:
        print(f"  [{model_name} @ {dataset}] saved in {elapsed:.1f}s -> {probe.describe()}")
    return probe


def extract_many(
    models: list[str],
    datasets: list[str],
    *,
    overwrite: bool = False,
    **kwargs,
) -> dict[tuple[str, str], str]:
    """Extract the full grid, skipping what already exists."""
    status: dict[tuple[str, str], str] = {}
    source_data = kwargs.get("source_data", config.SOURCE_DATA)
    kwargs.setdefault("resize_mode", config.RESIZE_MODE)
    for m in models:
        for d in datasets:
            d = config.canonical_dataset(d)
            if not overwrite and ProbeData.exists(m, d, source_data):
                status[(m, d)] = "cached"
                print(f"  [{m} @ {d}] cached")
                continue
            try:
                extract_probe(m, d, **kwargs)
                status[(m, d)] = "ok"
            except Exception as exc:
                status[(m, d)] = f"failed: {type(exc).__name__}: {exc}"
                print(f"  [{m} @ {d}] FAILED {type(exc).__name__}: {exc}")
    return status
