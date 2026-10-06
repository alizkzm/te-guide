"""`ProbeData` - everything a TE score is ever allowed to look at."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from guide import config
from guide.core.linalg import remap_labels


class MissingProbeError(FileNotFoundError):
    """Raised when a score needs a probe that has not been extracted yet."""


def _atomic_savez(path: Path, arrays: dict) -> None:
    """Write an .npz via a temp file + rename, so an interrupted job can never leave a half-written archive at the real path."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        np.savez_compressed(tmp, **arrays)
        written = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
        os.replace(written, path)
    finally:
        for leftover in (tmp, tmp.with_suffix(tmp.suffix + ".npz")):
            if leftover.exists():
                leftover.unlink()


def _is_readable_npz(path: Path) -> bool:
    """True when `path` is a complete, loadable probe archive."""
    try:
        with np.load(path) as z:
            return "features" in z.files and "labels" in z.files
    except Exception:
        return False


def probe_dir(model_name: str, source_data: str = config.SOURCE_DATA) -> Path:
    return config.PROBE_DIR / source_data / model_name


def probe_paths(model_name: str, dataset: str, source_data: str = config.SOURCE_DATA):
    d = probe_dir(model_name, source_data)
    return d / f"{dataset}.npz", d / f"{dataset}.json"


def weights_paths(model_name: str, source_data: str = config.SOURCE_DATA):
    d = probe_dir(model_name, source_data)
    return d / "_weights.npz", d / "_weights.json"


@dataclass
class ProbeData:
    """Cached forward-pass artefacts for one (model, target dataset) pair."""

    model_name: str
    dataset: str
    source_data: str = config.SOURCE_DATA

    features: np.ndarray | None = None
    labels: np.ndarray | None = None
    logits: np.ndarray | None = None

    weights: dict[str, np.ndarray] = field(default_factory=dict)

    meta: dict = field(default_factory=dict)

    source_probe: "ProbeData | None" = None

    @property
    def n_samples(self) -> int:
        return 0 if self.features is None else int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return 0 if self.features is None else int(self.features.shape[1])

    @property
    def n_classes(self) -> int:
        return 0 if self.labels is None else int(np.unique(self.labels).size)

    @property
    def has_head(self) -> bool:
        return self.logits is not None

    @property
    def probs(self) -> np.ndarray:
        """Softmax of the source head over the target images (LEEP's p(z|x))."""
        if self.logits is None:
            raise ValueError(
                f"{self.model_name} was loaded head-less: no source logits available"
            )
        from guide.core.linalg import softmax

        return softmax(self.logits.astype(np.float64))

    @property
    def source_pseudo_labels(self) -> np.ndarray:
        """argmax of the source head - the `z` variable of NCE / OTCE."""
        return self.logits.argmax(axis=1).astype(np.int64)

    def describe(self) -> str:
        return (
            f"{self.model_name} @ {self.dataset}: N={self.n_samples} D={self.feature_dim} "
            f"C={self.n_classes} head={'yes' if self.has_head else 'no'}"
        )

    def subset(self, idx: np.ndarray) -> "ProbeData":
        """A view restricted to `idx` (weights / meta / source probe are shared)."""
        meta = self.meta
        n_train = meta.get("n_train")
        if n_train is not None:
            meta = {**meta, "n_train": int(np.count_nonzero(np.asarray(idx) < int(n_train)))}
        return ProbeData(
            model_name=self.model_name,
            dataset=self.dataset,
            source_data=self.source_data,
            features=None if self.features is None else self.features[idx],
            labels=None if self.labels is None else remap_labels(self.labels[idx]),
            logits=None if self.logits is None else self.logits[idx],
            weights=self.weights,
            meta=meta,
            source_probe=self.source_probe,
        )

    def save(self) -> None:
        npz_path, json_path = probe_paths(self.model_name, self.dataset, self.source_data)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"features": self.features, "labels": self.labels}
        if self.logits is not None:
            arrays["logits"] = self.logits
        _atomic_savez(npz_path, {k: v for k, v in arrays.items() if v is not None})
        json_path.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

        if self.weights:
            w_npz, w_json = weights_paths(self.model_name, self.source_data)
            if not w_npz.exists():
                _atomic_savez(w_npz, self.weights)
                w_json.write_text(
                    json.dumps({k: list(v.shape) for k, v in self.weights.items()}, indent=2),
                    encoding="utf-8",
                )

    @classmethod
    def load(
        cls,
        model_name: str,
        dataset: str,
        source_data: str = config.SOURCE_DATA,
        with_weights: bool = True,
    ) -> "ProbeData":
        npz_path, json_path = probe_paths(model_name, dataset, source_data)
        if not npz_path.exists():
            raise MissingProbeError(
                f"no probe for ({model_name}, {dataset}) at {npz_path}.\n"
                f"    run:  python run.py extract -m {model_name} -d {dataset}"
            )
        try:
            with np.load(npz_path) as z:
                features = z["features"].astype(np.float64)
                labels = remap_labels(z["labels"])
                logits = z["logits"].astype(np.float64) if "logits" in z.files else None
        except Exception as exc:
            size = npz_path.stat().st_size / 1e6 if npz_path.exists() else 0
            raise MissingProbeError(
                f"probe for ({model_name}, {dataset}) is unreadable: "
                f"{type(exc).__name__}: {exc}\n"
                f"    {npz_path} ({size:.1f} MB) - most likely a job killed or a "
                f"disk filled mid-write. Delete it and re-extract:\n"
                f"    python tools/check_probes.py --delete\n"
                f"    python run.py extract -m {model_name} -d {dataset}"
            ) from exc

        meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}

        weights: dict[str, np.ndarray] = {}
        if with_weights:
            w_npz, _ = weights_paths(model_name, source_data)
            if w_npz.exists():
                with np.load(w_npz) as z:
                    weights = {k: z[k].astype(np.float64) for k in z.files}

        return cls(
            model_name=model_name,
            dataset=dataset,
            source_data=source_data,
            features=features,
            labels=labels,
            logits=logits,
            weights=weights,
            meta=meta,
        )

    @classmethod
    def exists(cls, model_name: str, dataset: str, source_data: str = config.SOURCE_DATA) -> bool:
        """True only for a *readable* probe."""
        path = probe_paths(model_name, dataset, source_data)[0]
        return path.exists() and _is_readable_npz(path)

    def attach_source_probe(
        self, dataset: str = config.SOURCE_PROBE_DATASET, required: bool = True
    ) -> "ProbeData | None":
        """Load the source-domain probe used by source-dependent scores."""
        if self.source_probe is not None:
            return self.source_probe
        try:
            self.source_probe = ProbeData.load(
                self.model_name, dataset, self.source_data, with_weights=False
            )
        except MissingProbeError:
            if required:
                raise MissingProbeError(
                    f"score needs the source-domain probe '{dataset}' for "
                    f"{self.model_name}.\n"
                    f"    run:  python run.py extract -m {self.model_name} -d {dataset}"
                ) from None
            self.source_probe = None
        return self.source_probe


def synthetic_probe(
    model_name: str = "synthetic_net",
    dataset: str = "synthetic",
    n_samples: int = 400,
    feature_dim: int = 48,
    n_classes: int = 6,
    n_source_classes: int = 25,
    separability: float = 0.6,
    with_head: bool = True,
    seed: int = 0,
) -> ProbeData:
    """A fake but well-formed probe, good enough to exercise every score."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_classes, size=n_samples)
    centres = rng.randn(n_classes, feature_dim) * separability
    features = centres[y] + rng.randn(n_samples, feature_dim)

    logits = None
    weights = {}
    if with_head:
        W_c = rng.randn(n_source_classes, feature_dim) / np.sqrt(feature_dim)
        b_c = rng.randn(n_source_classes) * 0.01
        logits = features @ W_c.T + b_c
        weights = {
            "W_classifier": W_c,
            "W_penultimate": rng.randn(feature_dim, feature_dim) / np.sqrt(feature_dim),
        }

    return ProbeData(
        model_name=model_name,
        dataset=dataset,
        features=features,
        labels=y.astype(np.int64),
        logits=logits,
        weights=weights,
        meta={
            "n_params": 1_000_000 + seed * 7919,
            "flops": None,
            "feature_dim": feature_dim,
            "n_source_classes": n_source_classes,
            "source_top1": 70.0 + (seed % 10),
            "synthetic": True,
        },
    )
