#!/usr/bin/env python
"""GUIDE command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from guide import config, ground_truth as gt_mod
from guide.core.registry import REGISTRY, get_score, list_scores


def _models(arg: str) -> list[str]:
    return config.resolve_hub(arg)


def _datasets(arg: str) -> list[str]:
    if arg.strip().lower() in ("all", "*"):
        return list(config.TARGET_DATASETS)
    return [config.canonical_dataset(d) for d in arg.split(",") if d.strip()]


def _scores(arg: str) -> list[str]:
    a = arg.strip().lower()
    if a in ("all", "*"):
        return list_scores()
    if a in ("te", "trivial", "label_free", "ready", "partial"):
        return list_scores(a)
    if a == "complete":
        return list_scores("ready")
    if a == "incomplete":
        return list_scores("partial")
    return [s.strip() for s in arg.split(",") if s.strip()]


def _hparams(pairs: list[str] | None) -> dict[str, dict]:
    """`--set etran.alpha=0.3 --set itm.train_iter=200` -> {"etran": {...}, ...}"""
    out: dict[str, dict] = {}
    for item in pairs or []:
        key, _, value = item.partition("=")
        score, _, param = key.partition(".")
        if not param:
            raise SystemExit(f"--set expects score.param=value, got {item!r}")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        out.setdefault(score, {})[param] = parsed
    return out


def cmd_list(args):
    rows = []
    for name in list_scores(include_disabled=True):
        cls = REGISTRY[name]
        s = cls()
        rows.append(
            (
                name,
                s.kind,
                s.completeness,
                "yes" if s.label_free else "no",
                ",".join(s.elements) or name,
                s.paper,
            )
        )
    print(f"{'score':18s} {'kind':8s} {'status':11s} {'label-free':11s} elements")
    print("-" * 110)
    for name, kind, status, lf, elements, paper in rows:
        flag = "  [DISABLED]" if name in config.DISABLED_SCORES else ""
        print(f"{name:18s} {kind:8s} {status:11s} {lf:11s} {elements}{flag}")
        print(f"{'':18s} {paper}")
    n_te = sum(1 for r in rows if r[1] == "TE")
    print(f"\n{len(rows)} scores: {n_te} TE + {len(rows) - n_te} trivial;  "
          f"{sum(1 for r in rows if r[2] == 'incomplete')} incomplete")
    if config.DISABLED_SCORES:
        print(f"disabled (skipped by '-s all', run by name to force): "
              f"{', '.join(sorted(config.DISABLED_SCORES))}")


def cmd_selftest(args):
    from guide.core.probe import synthetic_probe
    from guide.scores.base import ScoreNotApplicable

    probe = synthetic_probe(
        n_samples=args.n, feature_dim=args.d, n_classes=args.c, seed=1
    )
    probe.source_probe = synthetic_probe(
        dataset=config.SOURCE_PROBE_DATASET, n_samples=args.n, feature_dim=args.d,
        n_classes=20, seed=2,
    )
    names = _scores(args.scores)
    if args.fast:
        names = [n for n in names if n != "itm"]

    ok = skipped = failed = 0
    for name in names:
        score = get_score(name)
        try:
            out = score.run(probe, **_hparams(args.set).get(name, {}))
            vals = ", ".join(f"{k}={v:.4g}" for k, v in out.elements.items())
            print(f"  {name:18s} OK      {out.run_time:6.2f}s  {vals}")
            ok += 1
        except ScoreNotApplicable as exc:
            print(f"  {name:18s} SKIP    {str(exc).splitlines()[0][:70]}")
            skipped += 1
        except Exception as exc:
            print(f"  {name:18s} FAIL    {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{ok} ok, {skipped} skipped (incomplete by design), {failed} failed")
    return 1 if failed else 0


def cmd_mock(args):
    """Write synthetic probes so score/post/evaluate can be exercised offline."""
    from guide.core.probe import synthetic_probe
    from guide.ground_truth import get_ground_truth

    config.ensure_dirs()
    gt = get_ground_truth(args.gt)
    models = _models(args.models)
    datasets = [d for d in _datasets(args.datasets) if gt.get(d)]

    for d in datasets:
        accs = gt[d]
        lo, hi = min(accs.values()), max(accs.values())
        for i, m in enumerate(models):
            if m not in accs:
                continue
            frac = (accs[m] - lo) / (hi - lo) if hi > lo else 0.5
            for ds, seed in ((d, i), (config.SOURCE_PROBE_DATASET, 1000 + i)):
                p = synthetic_probe(
                    model_name=m,
                    dataset=ds,
                    n_samples=args.n,
                    feature_dim=args.dim,
                    n_classes=args.c,
                    separability=0.2 + 0.8 * frac,
                    seed=seed,
                )
                p.meta["source_top1"] = 70.0 + 8.0 * frac
                p.meta["n_params"] = int(1e6 * (1 + 4 * frac))
                p.save()
    print(f"wrote synthetic probes for {len(models)} models x {len(datasets)} datasets "
          f"(+ the {config.SOURCE_PROBE_DATASET} source probe) to {config.PROBE_DIR}")


def cmd_extract(args):
    from guide.hub.extract import extract_many

    config.ensure_dirs()
    models, datasets = _models(args.models), _datasets(args.datasets)
    source = config.SOURCE_PROBE_DATASET
    datasets = [d for d in datasets if d != source]
    print(f"extracting {len(models)} models x {len(datasets) + int(args.with_source)} datasets")

    status = {}
    if args.with_source:
        status.update(extract_many(
            models, [source],
            overwrite=args.overwrite,
            split="val" if source == "imagenet" else "train",
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.workers,
            max_samples=args.max_samples or config.SOURCE_PROBE_SAMPLES,
            resize_mode=args.resize,
            with_flops=args.flops,
        ))

    status.update(extract_many(
        models,
        datasets,
        overwrite=args.overwrite,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_samples=args.max_samples,
        resize_mode=args.resize,
        with_flops=args.flops,
    ))
    bad = {k: v for k, v in status.items() if v.startswith("failed")}
    print(f"\n{sum(1 for v in status.values() if v == 'ok')} extracted, "
          f"{sum(1 for v in status.values() if v == 'cached')} cached, {len(bad)} failed")
    for (m, d), why in bad.items():
        print(f"  {m} @ {d}: {why}")


def cmd_score(args):
    from guide.runner import run_grid

    config.ensure_dirs()
    skipped = config.DISABLED_SCORES & set(list_scores(include_disabled=True))
    if skipped and args.scores.strip().lower() in ("all", "*", "te", "trivial",
                                                   "label_free", "ready", "partial",
                                                   "complete", "incomplete"):
        print(f"skipping disabled scores: {', '.join(sorted(skipped))} "
              f"(config.DISABLED_SCORES)")
    if args.variance and args.richardson:
        raise SystemExit("--variance and --richardson are mutually exclusive")
    plugin = "variance" if args.variance else "richardson" if args.richardson else None
    if plugin:
        print(f"resampling plugin: {plugin} (only feature+label single-value "
              f"scores are affected; others run normally)")
    summary = run_grid(
        _models(args.models),
        _datasets(args.datasets),
        _scores(args.scores),
        samples_per_class=(
            "all" if str(args.samples_per_class).lower() == "all" else int(args.samples_per_class)
        ),
        seed=args.seed,
        overwrite=args.overwrite,
        hparams=_hparams(args.set),
        variance=args.variance,
        richardson=args.richardson,
    )
    print(f"\n{summary}")


def cmd_post(args):
    from guide.postprocess import load_recipes, postprocess_all

    config.ensure_dirs()
    recipes = load_recipes(args.recipes)
    for item in args.set or []:
        key, _, value = item.partition("=")
        score, _, param = key.partition(".")
        if score in recipes and param:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            recipes[score] = recipes[score].with_params(**{param: value})
    models = _models(args.models) if args.models else None
    postprocess_all(
        _datasets(args.datasets), _scores(args.scores), models=models, recipes=recipes
    )


def cmd_evaluate(args):
    from guide.evaluate import evaluate, format_summary, per_dataset_table

    config.ensure_dirs()
    table = gt_mod.get_ground_truth(args.gt)
    models = _models(args.models) if args.models else gt_mod.hub_for(table)
    result = evaluate(
        _datasets(args.datasets) if args.datasets else None,
        _scores(args.scores),
        ground_truth=args.gt,
        models=models,
        metrics=[m.strip() for m in args.metrics.split(",")] if args.metrics else None,
    )
    print(format_summary(result, metric=args.sort_by, top=args.top))
    if args.detail:
        print()
        print(per_dataset_table(result, args.detail, metric=args.sort_by))
    print(f"\nwritten to {config.EVAL_DIR / config.SOURCE_DATA}")


def cmd_all(args):
    cmd_extract(args)
    cmd_score(args)
    cmd_post(args)
    cmd_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, with_models=True):
        if with_models:
            sp.add_argument("-m", "--models", default=config.DEFAULT_HUB,
                            help="hub name (cnn|cnn_ssl|vit|itm|all) or comma list")
        sp.add_argument("-d", "--datasets", default="all", help="'all' or comma list")
        sp.add_argument("-s", "--scores", default="all",
                        help="'all' | 'te' | 'trivial' | 'complete' | comma list")
        sp.add_argument("--set", action="append", metavar="score.param=value",
                        help="hyper-parameter override, repeatable")

    sp = sub.add_parser("list", help="list every registered score")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("selftest", help="run every score on a synthetic probe")
    sp.add_argument("-s", "--scores", default="all")
    sp.add_argument("--set", action="append", metavar="score.param=value")
    sp.add_argument("-n", type=int, default=400, help="samples")
    sp.add_argument("-d", type=int, default=48, help="feature dim")
    sp.add_argument("-c", type=int, default=6, help="classes")
    sp.add_argument("--fast", action="store_true", help="skip ITM (trains a net)")
    sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("mock", help="write synthetic probes (offline pipeline demo)")
    sp.add_argument("-m", "--models", default=config.DEFAULT_HUB)
    sp.add_argument("-d", "--datasets", default="cifar10,dtd")
    sp.add_argument("--gt", default=config.DEFAULT_GROUND_TRUTH)
    sp.add_argument("-n", type=int, default=400, help="samples per probe")
    sp.add_argument("--dim", type=int, default=48)
    sp.add_argument("-c", type=int, default=8, help="classes")
    sp.set_defaults(func=cmd_mock)

    sp = sub.add_parser("extract", help="forward passes -> probe cache")
    add_common(sp)
    sp.add_argument("--split", default="all", choices=["train", "val", "test", "trainval", "all"])
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    sp.add_argument("--workers", type=int, default=0)
    sp.add_argument("--max-samples", type=int, default=None,
                    help="cap images per dataset (useful for a smoke run)")
    sp.add_argument("--resize", default=config.RESIZE_MODE, choices=["squash", "center_crop"],
                    help="center_crop = Resize(S,bicubic)+CenterCrop(S), the SFDA/ETran/NCTI "
                         "protocol (default); squash = Resize((S,S))")
    sp.add_argument("--with-source", action="store_true", default=True,
                    help="also extract the source-domain probe (ImageNet val by default). "
                         "otce needs it; atc/ped use it when present.")
    sp.add_argument("--no-source", dest="with_source", action="store_false")
    sp.add_argument("--flops", action="store_true", help="also count FLOPs (needs fvcore/thop)")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("score", help="run scores -> {target}_{model}_{score}.json")
    add_common(sp)
    sp.add_argument("--samples-per-class", default=str(config.SAMPLES_PER_CLASS))
    sp.add_argument("--seed", type=int, default=config.SEED)
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--variance", action="store_true",
                    help="resampling plugin: bag every feature+label single-value "
                         "score over stratified subsamples (see guide/scores/resample.py). "
                         "Records file under the plain score name, so isolate the run "
                         "(GUIDE_ROOT=... or --overwrite) to avoid clobbering plain records.")
    sp.add_argument("--richardson", action="store_true",
                    help="resampling plugin: extrapolate every such score to n->inf "
                         "(2*A(N)-A(N/2)); same isolation caveat as --variance.")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("post", help="combine elements + normalise over the hub")
    sp.add_argument("-m", "--models", default=None)
    sp.add_argument("-d", "--datasets", default="all")
    sp.add_argument("-s", "--scores", default="all")
    sp.add_argument("--recipes", default=None, help="JSON file of recipe overrides")
    sp.add_argument("--set", action="append", metavar="score.param=value")
    sp.set_defaults(func=cmd_post)

    sp = sub.add_parser("evaluate", help="correlations vs the ground truth")
    sp.add_argument("-m", "--models", default=None,
                    help="default: exactly the hub of the chosen ground truth")
    sp.add_argument("-d", "--datasets", default=None)
    sp.add_argument("-s", "--scores", default="all")
    sp.add_argument("--gt", default=config.DEFAULT_GROUND_TRUTH,
                    help="sfda | sfda_cnn | sfda_cnn_head | sfda_cnn_ssl | sfda_vit | itm | path")
    sp.add_argument("--metrics", default=None, help="comma list; default all")
    sp.add_argument("--sort-by", default="weighted_kendall")
    sp.add_argument("--top", type=int, default=None)
    sp.add_argument("--detail", default=None, help="print the per-dataset table for this score")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("all", help="extract -> score -> post -> evaluate")
    add_common(sp)
    sp.add_argument("--split", default="all")
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    sp.add_argument("--workers", type=int, default=0)
    sp.add_argument("--max-samples", type=int, default=None)
    sp.add_argument("--resize", default=config.RESIZE_MODE, choices=["squash", "center_crop"])
    sp.add_argument("--with-source", action="store_true", default=True)
    sp.add_argument("--no-source", dest="with_source", action="store_false")
    sp.add_argument("--flops", action="store_true")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--samples-per-class", default=str(config.SAMPLES_PER_CLASS))
    sp.add_argument("--seed", type=int, default=config.SEED)
    sp.add_argument("--recipes", default=None)
    sp.add_argument("--gt", default=config.DEFAULT_GROUND_TRUTH)
    sp.add_argument("--metrics", default=None)
    sp.add_argument("--sort-by", default="weighted_kendall")
    sp.add_argument("--top", type=int, default=None)
    sp.add_argument("--detail", default=None)
    sp.set_defaults(func=cmd_all)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(args.func(args) or 0)
