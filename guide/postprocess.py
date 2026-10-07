"""Stage 3: turn raw score elements into one comparable number per model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from guide import config
from guide.core.registry import REGISTRY, _ensure_loaded, list_scores
from guide.core.schema import iter_records

def _normalise(values: dict[str, float], mode: str) -> dict[str, float]:
    mode = (mode or "none").lower()
    if mode == "none":
        return dict(values)

    keys = list(values)
    v = np.array([values[k] for k in keys], dtype=np.float64)
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return dict(values)

    out = np.full_like(v, np.nan)
    if mode == "minmax":
        lo, hi = v[finite].min(), v[finite].max()
        out[finite] = (v[finite] - lo) / (hi - lo) if hi > lo else 0.0
    elif mode == "zscore":
        mu, sd = v[finite].mean(), v[finite].std(ddof=0)
        out[finite] = (v[finite] - mu) / sd if sd > 0 else 0.0
    elif mode == "rank":
        from scipy.stats import rankdata

        r = rankdata(v[finite])
        out[finite] = (r - 1) / max(len(r) - 1, 1)
    elif mode == "rank1":
        from scipy.stats import rankdata

        out[finite] = rankdata(-v[finite])
    elif mode == "neg":
        out[finite] = -v[finite]
    else:
        raise ValueError(f"unknown normalisation {mode!r}")
    return {k: float(x) for k, x in zip(keys, out)}


@dataclass
class Recipe:
    """How to collapse one score's elements into one number per model."""

    score_name: str
    weights: dict[str, float] = field(default_factory=dict)
    element_norm: str = "none"
    final_norm: str = "none"
    average: bool = False
    derive: Callable[[dict[str, float]], dict[str, float]] | None = None
    params: dict = field(default_factory=dict)
    note: str = ""

    def with_params(self, **kw) -> "Recipe":
        r = Recipe(
            score_name=self.score_name,
            weights=dict(self.weights),
            element_norm=self.element_norm,
            average=self.average,
            final_norm=self.final_norm,
            derive=self.derive,
            params={**self.params, **kw},
            note=self.note,
        )
        r.rebuild_weights()
        return r

    def rebuild_weights(self) -> "Recipe":
        """Recompute `weights` from `params` for the parametric recipes."""
        p = self.params
        if self.score_name == "etran":
            a = float(p.get("alpha", 0.15))
            self.weights = {"energy": a, "lda": 1 - a}
        return self


DEFAULT_RECIPES: dict[str, Recipe] = {
    "etran": Recipe(
        "etran",
        weights={"energy": 0.15, "lda": 0.85},
        element_norm="minmax",
        params={"alpha": 0.15},
        note="reference combination from mgholamikn/ETran/tw.py; alpha weights energy",
    ),
    "ncti": Recipe(
        "ncti",
        weights={"cls_conf": 1.0, "feature_nuc": 1.0, "log_class_pred_nuc": -1.0},
        element_norm="minmax",
        note="reference combination: mcscore + mascore - cpscore",
    ),
    "pactran": Recipe(
        "pactran",
        weights={"pactran_gaussian": 1.0},
        note="the Gaussian prior is the variant reported in most TE benchmarks",
    ),
    "ped": Recipe(
        "ped",
        weights={"ped_sfda": 1.0},
        note="PED+SFDA is the combination reported in the paper",
    ),
    "sa": Recipe(
        "sa",
        weights={"sa_sfda": 1.0},
        note="SA+SFDA is the headline combination of the WACV 2025 paper",
    ),
    "atc": Recipe(
        "atc",
        weights={"atc_ne": 1.0},
        note="ATC-NE (negative entropy) is the stronger variant in Garg et al.",
    ),
    "rep_var": Recipe(
        "rep_var",
        weights={"rep_var_feature": -1.0, "rep_var_logit": -1.0},
        element_norm="zscore",
        note="higher representation variance implies worse transfer -> negated",
    ),
    "otce": Recipe(
        "otce",
        weights={"task_difference": -1.0},
        note="auxiliary-free OTCE (a.k.a. OT-based NCE); both terms are lower-is-better",
    ),
}


def default_recipe(score_name: str) -> Recipe:
    if score_name in DEFAULT_RECIPES:
        return DEFAULT_RECIPES[score_name]
    _ensure_loaded()
    cls = REGISTRY.get(score_name)
    elements = getattr(cls, "elements", ()) if cls else ()
    element = elements[0] if elements else score_name
    return Recipe(score_name, weights={element: 1.0})


def load_recipes(path: str | Path | None) -> dict[str, Recipe]:
    """Merge a JSON override file into the default recipe book."""
    book = {name: default_recipe(name) for name in list_scores()}
    if not path:
        return book
    override = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, spec in override.items():
        base = book.get(name) or default_recipe(name)
        if "params" in spec:
            base = base.with_params(**spec["params"])
        if "weights" in spec:
            base.weights = {k: float(v) for k, v in spec["weights"].items()}
        for key in ("element_norm", "final_norm", "note"):
            if key in spec:
                setattr(base, key, spec[key])
        book[name] = base
    return book


def combine_hub(
    raw: dict[str, dict[str, float]],
    recipe: Recipe,
    on_missing: str = "drop_element",
) -> tuple[dict[str, float], dict]:
    """`raw` = {model: {element: value}} -> ({model: score}, diagnostics)."""
    models = list(raw)
    if not models:
        return {}, {"reason": "no records"}

    table = {m: dict(raw[m]) for m in models}
    if recipe.derive is not None:
        import inspect

        accepted = set(inspect.signature(recipe.derive).parameters) - {"elements"}
        kw = {k: v for k, v in recipe.params.items() if k in accepted}
        for m in models:
            table[m].update(recipe.derive(table[m], **kw))

    dropped: list[str] = []
    usable: dict[str, float] = {}
    for el, w in recipe.weights.items():
        if w == 0:
            continue
        vals = [table[m].get(el) for m in models]
        ok = [v is not None and np.isfinite(v) for v in vals]
        if all(ok):
            usable[el] = w
        elif on_missing == "drop_element":
            dropped.append(el)
        else:
            usable[el] = w

    if not usable:
        return {m: float("nan") for m in models}, {
            "reason": "every weighted element missing across the hub",
            "dropped_elements": dropped,
        }

    normed = {
        el: _normalise({m: float(table[m].get(el, np.nan)) for m in models}, recipe.element_norm)
        for el in usable
    }
    denom = 1.0
    if recipe.average:
        tot = float(sum(usable.values()))
        denom = tot if abs(tot) > 1e-12 else 1.0
    combined = {
        m: float(sum(w * normed[el].get(m, np.nan) for el, w in usable.items())
                 / denom)
        for m in models
    }
    combined = _normalise(combined, recipe.final_norm)

    return combined, {
        "elements_used": sorted(usable),
        "weights": usable,
        "dropped_elements": dropped,
        "element_norm": recipe.element_norm,
        "final_norm": recipe.final_norm,
        "params": recipe.params,
    }


def collect_raw(
    target_data: str,
    score_name: str,
    source_data: str = config.SOURCE_DATA,
    models: list[str] | None = None,
    score_root: Path | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
    """Read every record for one (dataset, score) -> ({model: elements}, {model: meta})."""
    raw: dict[str, dict[str, float]] = {}
    info: dict[str, dict] = {}
    for rec in iter_records(source_data, target_data, score_name, score_root):
        m = rec["model_name"]
        if models is not None and m not in models:
            continue
        if rec.get("status") != "ok":
            info[m] = {"status": rec.get("status"), "error": rec.get("error")}
            continue
        raw[m] = {k: v for k, v in (rec.get("score_elements_list") or {}).items() if v is not None}
        info[m] = {
            "status": "ok",
            "run_time": rec.get("run_time"),
            "samples_per_class": rec.get("samples_per_class"),
            "n_samples": rec.get("n_samples"),
        }
    return raw, info


def postprocess_one(
    target_data: str,
    score_name: str,
    recipe: Recipe | None = None,
    *,
    source_data: str = config.SOURCE_DATA,
    models: list[str] | None = None,
    score_root: Path | None = None,
    out_root: Path | None = None,
    write: bool = True,
) -> dict:
    raw, info = collect_raw(target_data, score_name, source_data, models, score_root)
    recipe = recipe or default_recipe(score_name)
    values, diag = combine_hub(raw, recipe)

    payload = {
        "target_data": target_data,
        "score_name": score_name,
        "source_data": source_data,
        "values": values,
        "raw_elements": raw,
        "recipe": {
            "weights": recipe.weights,
            "element_norm": recipe.element_norm,
            "final_norm": recipe.final_norm,
            "params": recipe.params,
            "note": recipe.note,
        },
        "diagnostics": diag,
        "per_model_info": info,
        "total_run_time": float(
            sum(v.get("run_time") or 0.0 for v in info.values() if v.get("status") == "ok")
        ),
    }
    if write:
        out = Path(out_root or config.POST_DIR) / source_data
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{target_data}_{score_name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return payload


def postprocess_all(
    datasets: list[str],
    scores: list[str] | None = None,
    *,
    source_data: str = config.SOURCE_DATA,
    models: list[str] | None = None,
    recipes: dict[str, Recipe] | None = None,
    score_root: Path | None = None,
    out_root: Path | None = None,
    verbose: bool = True,
) -> dict[tuple[str, str], dict]:
    scores = scores or list_scores()
    recipes = recipes or {n: default_recipe(n) for n in scores}
    out: dict[tuple[str, str], dict] = {}
    for d in datasets:
        d = config.canonical_dataset(d)
        for s in scores:
            payload = postprocess_one(
                d,
                s,
                recipes.get(s),
                source_data=source_data,
                models=models,
                score_root=score_root,
                out_root=out_root,
            )
            n_ok = sum(1 for v in payload["values"].values() if np.isfinite(v))
            out[(d, s)] = payload
            if verbose and n_ok:
                print(f"  {d:12s} {s:14s} {n_ok:3d} models")
    return out
