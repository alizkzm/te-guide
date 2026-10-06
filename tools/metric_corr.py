#!/usr/bin/env python
"""Metric-vs-metric correlation matrix, computed on top of $GUIDE_ROOT/scores."""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations_with_replacement
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr, weightedtau

from guide import config
from guide.core.registry import list_scores
from guide.ground_truth import get_ground_truth
from guide.postprocess import collect_raw, combine_hub, default_recipe

CORRS = {
    "spearman": lambda a, b: float(spearmanr(a, b)[0]),
    "kendall": lambda a, b: float(kendalltau(a, b)[0]),
    "weighted_kendall": lambda a, b: float(weightedtau(a, b)[0]),
    "pearson": lambda a, b: float(pearsonr(a, b)[0]),
}


def load_values(scores, datasets, models, source):
    """values[score][dataset] -> {model: float} (only finite entries kept)."""
    values: dict[str, dict[str, dict[str, float]]] = {}
    for s in scores:
        recipe = default_recipe(s)
        per_ds = {}
        for d in datasets:
            raw, _ = collect_raw(d, s, source_data=source, models=models)
            if not raw:
                continue
            combined, _ = combine_hub(raw, recipe)
            finite = {m: float(v) for m, v in combined.items()
                      if v is not None and np.isfinite(v)}
            if finite:
                per_ds[d] = finite
        if per_ds:
            values[s] = per_ds
    return values


def pair_corr_per_dataset(va, vb, corr):
    """Mean of within-dataset correlations; also the count of usable datasets."""
    vals = []
    for d in set(va) & set(vb):
        shared = [m for m in va[d] if m in vb[d]]
        if len(shared) < 3:
            continue
        a = np.array([va[d][m] for m in shared], float)
        b = np.array([vb[d][m] for m in shared], float)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        c = corr(a, b)
        if np.isfinite(c):
            vals.append(c)
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def pair_corr_pooled(va, vb, corr):
    """Rank within each dataset, pool the ranks, correlate the pooled vectors."""
    a_all, b_all = [], []
    for d in set(va) & set(vb):
        shared = [m for m in va[d] if m in vb[d]]
        if len(shared) < 3:
            continue
        a = np.array([va[d][m] for m in shared], float)
        b = np.array([vb[d][m] for m in shared], float)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        a_all.append(rankdata(a))
        b_all.append(rankdata(b))
    if not a_all:
        return np.nan, 0
    a = np.concatenate(a_all)
    b = np.concatenate(b_all)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan, len(a)
    return corr(a, b), int(len(a))


def order_by_cluster(names, M):
    """Hierarchical leaf order on 1 - |corr|; identity order if scipy absent."""
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
    except Exception:
        return list(range(len(names)))
    D = 1.0 - np.abs(np.nan_to_num(M, nan=0.0))
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    try:
        Z = linkage(squareform(D, checks=False), method="average")
        return list(leaves_list(Z))
    except Exception:
        return list(range(len(names)))


def write_matrix(path, names, M):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([""] + names)
        for i, r in enumerate(names):
            w.writerow([r] + ["" if not np.isfinite(M[i, j]) else f"{M[i, j]:.4f}"
                              for j in range(len(names))])


def heatmap(path, names, M, corr_name, mode):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib not available - skipping heatmap)")
        return
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * n + 3), max(5, 0.55 * n + 2)), dpi=200)
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    if n <= 24:
        for i in range(n):
            for j in range(n):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(M[i, j]) > 0.6 else "#222")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"{corr_name} ({mode})", fontsize=9)
    ax.set_title(f"metric-metric {corr_name}, {mode}", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--scores", default="all",
                    help="comma list of metrics, or 'all' (default)")
    ap.add_argument("--gt", default=config.DEFAULT_GROUND_TRUTH,
                    help="ground truth whose hub/datasets set the defaults")
    ap.add_argument("-m", "--models", default=None, help="hub override (name or comma list)")
    ap.add_argument("-d", "--datasets", default=None, help="datasets override (comma list)")
    ap.add_argument("--source", default=config.SOURCE_DATA)
    ap.add_argument("--corr", default="spearman", choices=list(CORRS))
    ap.add_argument("--mode", default="per_dataset", choices=["per_dataset", "pooled"])
    ap.add_argument("--cluster", action="store_true",
                    help="order rows/cols by hierarchical clustering on 1-|corr|")
    ap.add_argument("--heatmap", action="store_true", help="also write a PNG heatmap")
    ap.add_argument("--out", default="runs_metric_corr")
    args = ap.parse_args()

    gt_table = get_ground_truth(args.gt)
    datasets = ([config.canonical_dataset(d) for d in args.datasets.split(",")]
                if args.datasets else [d for d in gt_table if gt_table[d]])
    models = (config.resolve_hub(args.models) if args.models
              else sorted({m for d in datasets for m in gt_table[d]}))
    requested = (list_scores() if args.scores in ("all", "*")
                 else [s.strip() for s in args.scores.split(",") if s.strip()])

    print(f"ground truth : {args.gt}")
    print(f"hub          : {len(models)} models   {len(datasets)} datasets")
    print(f"metrics asked: {len(requested)}")
    print(f"correlation  : {args.corr}   mode: {args.mode}\n")

    values = load_values(requested, datasets, models, args.source)
    scores = [s for s in requested if s in values]
    missing = [s for s in requested if s not in values]
    if missing:
        print(f"no records for {len(missing)} metric(s): {', '.join(missing)}\n")
    if len(scores) < 2:
        print("need at least 2 metrics with records - run `run.py score` first")
        return 1

    corr = CORRS[args.corr]
    pair = pair_corr_per_dataset if args.mode == "per_dataset" else pair_corr_pooled

    n = len(scores)
    M = np.full((n, n), np.nan)
    COV = np.zeros((n, n), dtype=int)
    for i, j in combinations_with_replacement(range(n), 2):
        c, cov = pair(values[scores[i]], values[scores[j]], corr)
        M[i, j] = M[j, i] = c
        COV[i, j] = COV[j, i] = cov

    order = order_by_cluster(scores, M) if args.cluster else list(range(n))
    names = [scores[k] for k in order]
    Mo = M[np.ix_(order, order)]
    COVo = COV[np.ix_(order, order)]

    out = (Path(__file__).resolve().parent.parent / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"metric_corr_{args.corr}_{args.mode}"
    write_matrix(out / f"{stem}.csv", names, Mo)
    write_matrix(out / f"{stem}_coverage.csv", names,
                 COVo.astype(float))
    if args.heatmap:
        heatmap(out / f"{stem}.png", names, Mo, args.corr, args.mode)

    w = max(len(s) for s in names) + 1
    print(f"{'':{w}}" + " ".join(f"{s[:6]:>6s}" for s in names))
    for i, r in enumerate(names):
        cells = " ".join("     ." if not np.isfinite(Mo[i, j]) else f"{Mo[i, j]:6.2f}"
                         for j in range(n))
        print(f"{r:{w}s}{cells}")

    pairs = [(names[i], names[j], Mo[i, j])
             for i in range(n) for j in range(i + 1, n) if np.isfinite(Mo[i, j])]
    if pairs:
        pairs.sort(key=lambda t: -t[2])
        print("\nmost correlated:")
        for a, b, c in pairs[:8]:
            print(f"  {c:+.3f}  {a} - {b}")
        print("least correlated:")
        for a, b, c in pairs[-8:][::-1]:
            print(f"  {c:+.3f}  {a} - {b}")

    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
