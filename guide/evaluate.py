"""Stage 4: correlate each post-processed score against the ground truth."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr, weightedtau

from guide import config
from guide.core.registry import REGISTRY, _ensure_loaded, list_scores
from guide.ground_truth import DEFAULT_GT, get_ground_truth


def weighted_pearson(x, y) -> float:
    """Pearson with linearly decaying weights in the order given (best first)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return float("nan")
    w = np.arange(n, 0, -1, dtype=float)
    mx = (w * x).sum() / w.sum()
    my = (w * y).sum() / w.sum()
    dx, dy = x - mx, y - my
    num = (w * dx * dy).sum()
    den = math.sqrt((w * dx * dx).sum() * (w * dy * dy).sum())
    return float(num / den) if den > 0 else float("nan")


def rel_at_k(score: np.ndarray, gt: np.ndarray, k: int) -> float:
    order = np.argsort(-score)[:k]
    return float(gt[order].max() / gt.max()) if gt.max() != 0 else float("nan")


def recall_at_k(score: np.ndarray, gt: np.ndarray, k: int) -> float:
    top1_pred = int(np.argmax(score))
    true_topk = set(np.argsort(-gt)[:k].tolist())
    return float(top1_pred in true_topk)


def top1_regret(score: np.ndarray, gt: np.ndarray) -> float:
    return float(gt.max() - gt[int(np.argmax(score))])


ALL_METRICS = [
    "pearson", "spearman", "kendall", "weighted_kendall", "weighted_pearson",
    "rel@1", "rel@3", "recall@1", "recall@3", "top1_regret",
]


def correlations(
    score: dict[str, float],
    gt: dict[str, float],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """All metrics for one (dataset, score) pair, over the shared models."""
    metrics = metrics or ALL_METRICS
    shared = [m for m in score if m in gt and np.isfinite(score[m]) and np.isfinite(gt[m])]
    out: dict[str, float] = {"n_models": float(len(shared))}
    if len(shared) < 3:
        return {**out, **{k: float("nan") for k in metrics}}

    s = np.array([score[m] for m in shared], dtype=float)
    g = np.array([gt[m] for m in shared], dtype=float)

    if np.ptp(s) == 0 or np.ptp(g) == 0:
        return {**out, **{k: float("nan") for k in metrics}}

    order = np.argsort(-s)

    values = {
        "pearson": lambda: float(pearsonr(s, g)[0]),
        "spearman": lambda: float(spearmanr(s, g)[0]),
        "kendall": lambda: float(kendalltau(s, g)[0]),
        "weighted_kendall": lambda: float(weightedtau(s, g)[0]),
        "weighted_pearson": lambda: weighted_pearson(s[order], g[order]),
        "rel@1": lambda: rel_at_k(s, g, 1),
        "rel@3": lambda: rel_at_k(s, g, 3),
        "recall@1": lambda: recall_at_k(s, g, 1),
        "recall@3": lambda: recall_at_k(s, g, 3),
        "top1_regret": lambda: top1_regret(s, g),
    }
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in metrics:
            try:
                out[name] = values[name]()
            except Exception:
                out[name] = float("nan")
    return out


def _load_postprocessed(
    dataset: str, score_name: str, source_data: str, root: Path | None
) -> dict | None:
    p = Path(root or config.POST_DIR) / source_data / f"{dataset}_{score_name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate(
    datasets: list[str] | None = None,
    scores: list[str] | None = None,
    *,
    ground_truth: str = DEFAULT_GT,
    source_data: str = config.SOURCE_DATA,
    models: list[str] | None = None,
    metrics: list[str] | None = None,
    post_root: Path | None = None,
    out_root: Path | None = None,
    write: bool = True,
) -> dict:
    """Returns {"per_dataset": [...rows...], "summary": [...rows...]}."""
    _ensure_loaded()
    gt_table = get_ground_truth(ground_truth)
    datasets = [config.canonical_dataset(d) for d in (datasets or list(gt_table))]
    datasets = [d for d in datasets if gt_table.get(d)]
    scores = scores or list_scores()
    metrics = metrics or ALL_METRICS

    rows: list[dict] = []
    for score_name in scores:
        cls = REGISTRY.get(score_name)
        for d in datasets:
            payload = _load_postprocessed(d, score_name, source_data, post_root)
            if payload is None:
                continue
            values = {k: v for k, v in payload["values"].items() if v is not None}
            if models is not None:
                values = {k: v for k, v in values.items() if k in models}
            res = correlations(values, gt_table[d], metrics)
            rows.append(
                {
                    "score": score_name,
                    "kind": "TRIVIAL" if getattr(cls, "trivial", False) else "TE",
                    "completeness": (
                        "complete" if getattr(cls, "status", "ready") == "ready" else "incomplete"
                    ),
                    "label_free": bool(getattr(cls, "label_free", False)),
                    "dataset": d,
                    "run_time": payload.get("total_run_time"),
                    **res,
                }
            )

    summary: list[dict] = []
    for score_name in scores:
        sub = [r for r in rows if r["score"] == score_name and r["n_models"] >= 3]
        if not sub:
            continue
        entry = {
            "score": score_name,
            "kind": sub[0]["kind"],
            "completeness": sub[0]["completeness"],
            "label_free": sub[0]["label_free"],
            "n_datasets": len(sub),
            "total_run_time": float(np.nansum([r["run_time"] or 0.0 for r in sub])),
        }
        for m in metrics:
            vals = [r[m] for r in sub if r.get(m) is not None and np.isfinite(r[m])]
            entry[m] = float(np.mean(vals)) if vals else float("nan")
        summary.append(entry)
    summary.sort(key=lambda r: (-(r.get("weighted_kendall") or -9), r["score"]))

    result = {
        "ground_truth": ground_truth,
        "source_data": source_data,
        "datasets": datasets,
        "hub": sorted({m for d in datasets for m in gt_table[d]}),
        "per_dataset": rows,
        "summary": summary,
    }

    if write:
        out = Path(out_root or config.EVAL_DIR) / source_data
        out.mkdir(parents=True, exist_ok=True)
        tag = _slug(ground_truth)
        (out / f"evaluation_{tag}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        _write_csv(out / f"per_dataset_{tag}.csv", rows)
        _write_csv(out / f"summary_{tag}.csv", summary)
    return result


def _slug(name: str) -> str:
    """A ground truth may be a path; keep only a filename-safe stem."""
    stem = Path(str(name)).stem if any(c in str(name) for c in "/\\") else str(name)
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def format_summary(result: dict, metric: str = "weighted_kendall", top: int | None = None) -> str:
    rows = result["summary"]
    rows = sorted(rows, key=lambda r: -(r.get(metric) if np.isfinite(r.get(metric, np.nan)) else -9))
    if top:
        rows = rows[:top]
    datasets = result["datasets"]
    lines = [
        f"ground truth: {result['ground_truth']}   "
        f"hub: {len(result['hub'])} models   datasets: {len(datasets)}",
        "",
        f"{'score':18s} {'kind':8s} {'cmp':11s} {'LF':3s} {metric:>17s} "
        f"{'pearson':>9s} {'kendall':>9s} {'rel@1':>7s} {'rec@3':>7s} {'sec':>8s}",
        "-" * 106,
    ]
    for r in rows:
        lines.append(
            f"{r['score']:18s} {r['kind']:8s} {r['completeness']:11s} "
            f"{'y' if r['label_free'] else 'n':3s} "
            f"{r.get(metric, float('nan')):17.4f} "
            f"{r.get('pearson', float('nan')):9.4f} "
            f"{r.get('kendall', float('nan')):9.4f} "
            f"{r.get('rel@1', float('nan')):7.4f} "
            f"{r.get('recall@3', float('nan')):7.3f} "
            f"{r.get('total_run_time', 0.0):8.1f}"
        )
    return "\n".join(lines)


def per_dataset_table(result: dict, score_name: str, metric: str = "weighted_kendall") -> str:
    rows = [r for r in result["per_dataset"] if r["score"] == score_name]
    lines = [f"{score_name} - {metric}", "-" * 34]
    for r in sorted(rows, key=lambda r: r["dataset"]):
        lines.append(f"  {r['dataset']:14s} {r.get(metric, float('nan')):8.4f}  (n={int(r['n_models'])})")
    vals = [r[metric] for r in rows if np.isfinite(r.get(metric, np.nan))]
    if vals:
        lines.append(f"  {'MEAN':14s} {np.mean(vals):8.4f}")
    return "\n".join(lines)
