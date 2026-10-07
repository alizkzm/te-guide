"""Stage 2: run scores over the (model x dataset x score) grid -> JSON records."""

from __future__ import annotations

import json
import traceback

import numpy as np

from guide import config
from guide.core.probe import MissingProbeError, ProbeData
from guide.core.registry import get_score, list_scores
from guide.core.sampling import stratified_indices
from guide.core.schema import ScoreRecord, record_path, save_record
from guide.scores.base import ScoreNotApplicable


def run_one(
    probe: ProbeData,
    score_name: str,
    samples_per_class: str | int = config.SAMPLES_PER_CLASS,
    seed: int = config.SEED,
    hparams: dict | None = None,
) -> ScoreRecord:
    """Compute one score on one probe and return the (unsaved) record."""
    score = get_score(score_name)
    hparams = {**(hparams or {})}

    view = probe
    idx = stratified_indices(probe.labels, samples_per_class, seed=seed)
    if idx is not None:
        view = probe.subset(idx)
        view.source_probe = probe.source_probe

    rec = ScoreRecord(
        target_data=probe.dataset,
        model_name=probe.model_name,
        score_name=score_name,
        source_data=probe.source_data,
        samples_per_class=samples_per_class,
        n_samples=view.n_samples,
        n_classes=view.n_classes,
        feature_dim=view.feature_dim,
        seed=seed,
        score_element_names=list(score.elements),
        score_kind=score.kind,
        score_completeness=score.completeness,
        score_paper=score.paper,
        score_note=score.note,
        label_free=bool(score.label_free),
    )

    try:
        out = score.run(view, **(hparams or {}))
        rec.score_elements_list = out.elements
        rec.score_element_names = list(out.elements)
        rec.run_time = out.run_time
        rec.hyperparameters = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in out.hyperparameters.items()
        }
    except (ScoreNotApplicable, MissingProbeError) as exc:
        rec.status = "skipped"
        rec.error = str(exc)
    except Exception as exc:
        rec.status = "failed"
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.hyperparameters = {"traceback": traceback.format_exc(limit=3)}
    return rec


def run_grid(
    models: list[str],
    datasets: list[str],
    scores: list[str] | None = None,
    *,
    source_data: str = config.SOURCE_DATA,
    samples_per_class: str | int = config.SAMPLES_PER_CLASS,
    seed: int = config.SEED,
    overwrite: bool = False,
    hparams: dict[str, dict] | None = None,
    verbose: bool = True,
) -> dict:
    """Full grid."""
    scores = scores or list_scores()
    hparams = hparams or {}
    needs_source = any(
        "source_probe" in getattr(get_score(s), "requires", ()) or s in ("ped",)
        for s in scores
    )

    summary = {"ok": 0, "skipped": 0, "failed": 0, "cached": 0}
    for model in models:
        for dataset in datasets:
            dataset = config.canonical_dataset(dataset)

            todo, stale = [], []
            for s in scores:
                path = record_path(dataset, model, s, source_data)
                if overwrite or not path.exists():
                    todo.append(s)
                    continue
                try:
                    cached = json.loads(path.read_text(encoding="utf-8")).get("samples_per_class")
                except Exception:
                    cached = None
                if cached is not None and str(cached) != str(samples_per_class):
                    stale.append((s, cached))

            if stale and verbose:
                print(
                    f"  [{model} @ {dataset}] {len(stale)} cached record(s) were computed with a "
                    f"different samples_per_class (e.g. {stale[0][0]}: {stale[0][1]!r}, "
                    f"you asked for {samples_per_class!r}) and were NOT recomputed.\n"
                    f"      Use --overwrite, or isolate the sweep:\n"
                    f"      GUIDE_ROOT=runs_spc{samples_per_class} GUIDE_PROBES={config.PROBE_DIR} "
                    f"python run.py score --samples-per-class {samples_per_class}"
                )

            if not todo:
                summary["cached"] += len(scores)
                continue

            try:
                probe = ProbeData.load(model, dataset, source_data)
            except MissingProbeError as exc:
                if verbose:
                    lines = str(exc).splitlines()
                    print(f"  [{model} @ {dataset}] SKIPPED: {lines[0]}")
                    for extra in lines[1:]:
                        print(f"      {extra.strip()}")
                summary["skipped"] += len(todo)
                continue

            if needs_source:
                probe.attach_source_probe(required=False)

            summary["cached"] += len(scores) - len(todo)
            for s in todo:
                rec = run_one(
                    probe,
                    s,
                    samples_per_class=samples_per_class,
                    seed=seed,
                    hparams=hparams.get(s),
                )
                save_record(rec)
                summary[rec.status] = summary.get(rec.status, 0) + 1
                if verbose:
                    if rec.status == "ok":
                        vals = ", ".join(
                            f"{k}={v:.4g}" if v is not None and np.isfinite(v) else f"{k}=nan"
                            for k, v in rec.score_elements_list.items()
                        )
                        print(f"  {dataset:12s} {model:15s} {s:14s} {rec.run_time:7.2f}s  {vals}")
                    else:
                        head = (rec.error or "").splitlines()[0][:80]
                        print(f"  {dataset:12s} {model:15s} {s:14s} {rec.status.upper()}: {head}")
    return summary
