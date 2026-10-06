#!/usr/bin/env python
"""Data-free weight-spectrum statistics of each model, written for the w_* scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch.nn as nn

from guide import config
from guide.hub.models import load_model


def layer_matrices(model: nn.Module, min_dim: int):
    """Yield each Conv/Linear weight as a 2-D numpy matrix (out, fan_in)."""
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            W = mod.weight.detach().cpu().float().numpy()
            W = W.reshape(W.shape[0], -1)
            if min(W.shape) >= min_dim:
                yield W


def powerlaw_alpha(eig: np.ndarray) -> float:
    """Clauset MLE power-law exponent of the eigenvalue tail, xmin by min-KS."""
    e = np.sort(eig[eig > 0])
    if e.size < 8:
        return np.nan
    lo = int(0.10 * e.size)
    hi = max(lo + 1, int(0.90 * e.size))
    cand = np.unique(e[lo:hi])
    if cand.size > 60:
        cand = cand[np.linspace(0, cand.size - 1, 60).astype(int)]
    best_ks, best_a = np.inf, np.nan
    for xmin in cand:
        tail = e[e >= xmin]
        n = tail.size
        if n < 5:
            continue
        s = np.sum(np.log(tail / xmin))
        if s <= 0:
            continue
        a = 1.0 + n / s
        cdf_emp = np.arange(1, n + 1) / n
        cdf_fit = 1.0 - (tail / xmin) ** (1.0 - a)
        ks = np.max(np.abs(cdf_emp - cdf_fit))
        if ks < best_ks:
            best_ks, best_a = ks, a
    return float(best_a)


def mp_softrank(eig: np.ndarray, shape) -> float:
    """Fraction of the spectrum inside the Marchenko-Pastur bulk, lambda+/lambda_max."""
    lam_max = float(eig.max())
    if lam_max <= 0:
        return np.nan
    q = min(shape) / max(shape)
    sigma2 = float(np.mean(eig))
    lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
    return float(np.clip(lam_plus / lam_max, 0.0, 1.0))


def analyse(model: nn.Module, min_dim: int) -> dict:
    """Per-layer spectral statistics, averaged over the layers of the model."""
    per = {k: [] for k in ("alpha", "alpha_weighted", "log_spectral_norm",
                           "log_frobenius", "stable_rank", "effective_rank",
                           "mp_softrank")}
    n_layers = 0
    for W in layer_matrices(model, min_dim):
        s = np.linalg.svd(W, compute_uv=False)
        s = s[s > 0]
        if s.size < 2:
            continue
        eig = s ** 2
        lam_max = float(eig.max())
        p = s / s.sum()
        alpha = powerlaw_alpha(eig)
        per["alpha"].append(alpha)
        per["alpha_weighted"].append(alpha * np.log10(lam_max) if np.isfinite(alpha) else np.nan)
        per["log_spectral_norm"].append(float(np.log10(np.sqrt(lam_max))))
        per["log_frobenius"].append(float(0.5 * np.log10(eig.sum())))
        per["stable_rank"].append(float(eig.sum() / lam_max))
        per["effective_rank"].append(float(np.exp(-np.sum(p * np.log(p)))))
        per["mp_softrank"].append(mp_softrank(eig, W.shape))
        n_layers += 1
    out = {k: float(np.nanmean(v)) if any(np.isfinite(v)) else float("nan")
           for k, v in per.items()}
    out["n_layers"] = n_layers
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--models", default=config.DEFAULT_HUB,
                    help="hub name (sfda_cnn, cnn, all, ...) or comma-separated models")
    ap.add_argument("--min-dim", type=int, default=10,
                    help="skip weight matrices whose smaller dimension is below this")
    ap.add_argument("--out", default=None, help="output folder (default $GUIDE_ROOT/weightstats)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    models = config.resolve_hub(args.models)
    out = Path(args.out).resolve() if args.out else config.WEIGHTSTATS_DIR
    out.mkdir(parents=True, exist_ok=True)
    print(f"weights version : {config.WEIGHT_VERSION}")
    print(f"models          : {len(models)}")
    print(f"writing to      : {out}\n")
    print(f"{'model':16s} {'lyrs':>4s} {'alpha':>6s} {'a_wtd':>7s} {'logSN':>6s} "
          f"{'srank':>6s} {'erank':>6s} {'mpsr':>5s}")
    print("-" * 66)

    for name in models:
        path = out / f"{name}.json"
        if path.exists() and not args.overwrite:
            print(f"{name:16s}  exists (use --overwrite)")
            continue
        try:
            model = load_model(name).eval()
        except Exception as exc:
            print(f"{name:16s}  load failed: {exc}")
            continue
        stats = analyse(model, args.min_dim)
        stats["model_name"] = name
        stats["weights_version"] = config.WEIGHT_VERSION
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"{name:16s} {stats['n_layers']:4d} {stats['alpha']:6.2f} "
              f"{stats['alpha_weighted']:7.2f} {stats['log_spectral_norm']:6.2f} "
              f"{stats['stable_rank']:6.2f} {stats['effective_rank']:6.1f} "
              f"{stats['mp_softrank']:5.2f}")

    print(f"\ndone -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
