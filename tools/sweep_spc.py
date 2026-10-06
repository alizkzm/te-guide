#!/usr/bin/env python
"""Sweep `samples-per-class`, reusing one probe cache, isolating every output."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def run(cmd: list[str], env: dict, label: str, quiet: bool) -> bool:
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, env=env, cwd=HERE, text=True,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
    dt = time.perf_counter() - t0
    ok = proc.returncode == 0
    print(f"    {label:38s} {'ok ' if ok else 'FAILED'} {dt:7.1f}s")
    if not ok and quiet and proc.stdout:
        print("      " + "\n      ".join(proc.stdout.strip().splitlines()[-8:]))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spc", default="1,5,25,all",
                    help="comma list of samples-per-class settings")
    ap.add_argument("--probes", default="runs/probes",
                    help="the probe cache to reuse - never written to")
    ap.add_argument("--weightstats", default=None,
                    help="folder holding <model>.json weight stats (for the w_* scores). "
                         "Needed because each setting gets a fresh GUIDE_ROOT, so the "
                         "default $GUIDE_ROOT/weightstats would be empty. Usually "
                         "<world>/weightstats; falls back to $GUIDE_WEIGHTSTATS.")
    ap.add_argument("-m", "--models", default="cnn")
    ap.add_argument("-d", "--datasets", default="all")
    ap.add_argument("-s", "--scores", default="all")
    ap.add_argument("--gt", default="sfda_cnn")
    ap.add_argument("--out", default="runs_sweep", help="where the collected CSVs go")
    ap.add_argument("--root-prefix", default="runs_spc")
    ap.add_argument("--metric", default="weighted_kendall", help="metric for the pivot table")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--by-dataset", action="store_true",
                    help="also print one table per dataset, not just the mean")
    ap.add_argument("--verbose", action="store_true", help="stream each sub-command")
    args = ap.parse_args()

    probes = (HERE / args.probes).resolve()
    if not probes.is_dir():
        raise SystemExit(f"probe cache not found: {probes}\n"
                         f"    point --probes at the directory holding <source>/<model>/*.npz")
    n_probes = len(list(probes.rglob("*.npz")))
    settings = [s.strip() for s in args.spc.split(",") if s.strip()]

    wstats = None
    if args.weightstats:
        wstats = (HERE / args.weightstats).resolve()
        if not wstats.is_dir():
            raise SystemExit(f"weight stats not found: {wstats}\n"
                             f"    run `python tools/weight_probe.py -m <models>` for this world first")
    elif "GUIDE_WEIGHTSTATS" in os.environ:
        wstats = Path(os.environ["GUIDE_WEIGHTSTATS"]).resolve()

    print(f"probe cache : {probes}  ({n_probes} .npz, read-only)")
    print(f"weight stats: {wstats if wstats else 'not set - the w_* scores will be skipped'}")
    print(f"settings    : {settings}")
    print(f"hub/gt      : {args.models} / {args.gt}")
    print(f"roots       : {args.root_prefix}<spc>/   ->  collected in {args.out}/\n")

    roots: dict[str, Path] = {}
    for spc in settings:
        root = (HERE / f"{args.root_prefix}{spc}").resolve()
        roots[spc] = root
        env = {**os.environ, "GUIDE_PROBES": str(probes), "GUIDE_ROOT": str(root)}
        if wstats is not None:
            env["GUIDE_WEIGHTSTATS"] = str(wstats)
        print(f"[samples-per-class = {spc}]  -> {root.name}/")
        ok = run([sys.executable, "run.py", "score", "-m", args.models,
                  "-d", args.datasets, "-s", args.scores,
                  "--samples-per-class", spc, "--seed", str(args.seed)],
                 env, "score", not args.verbose)
        if ok:
            ok = run([sys.executable, "run.py", "post", "-d", args.datasets,
                      "-s", args.scores], env, "post", not args.verbose)
        if ok:
            run([sys.executable, "run.py", "evaluate", "--gt", args.gt,
                 "-s", args.scores], env, "evaluate", not args.verbose)
        print()

    out = (HERE / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, dataset_rows = [], []
    for spc, root in roots.items():
        for f in (root / "evaluation").rglob(f"evaluation_{args.gt}.json"):
            res = json.loads(f.read_text(encoding="utf-8"))
            for r in res.get("summary", []):
                summary_rows.append({"samples_per_class": spc, **r})
            for r in res.get("per_dataset", []):
                dataset_rows.append({"samples_per_class": spc, **r})

    if not summary_rows:
        print("no evaluation output found - did `score` succeed?")
        return 1

    def write(path: Path, rows: list[dict]) -> None:
        keys, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k); keys.append(k)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    write(out / "summary_by_spc.csv", summary_rows)
    write(out / "per_dataset.csv", dataset_rows)

    pivot: dict[str, dict[str, float]] = {}
    for r in summary_rows:
        pivot.setdefault(r["score"], {})[r["samples_per_class"]] = r.get(args.metric)
    last = settings[-1]
    order = sorted(pivot, key=lambda s: -(pivot[s].get(last) if isinstance(
        pivot[s].get(last), (int, float)) else -9))

    with (out / f"pivot_{args.metric}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["score", *settings])
        for s in order:
            w.writerow([s, *[pivot[s].get(k, "") for k in settings]])

    def table(title: str, cells: dict[str, dict[str, float]], keys: list[str]) -> None:
        rows = sorted(cells, key=lambda s: -(cells[s].get(keys[-1]) if isinstance(
            cells[s].get(keys[-1]), (int, float)) else -9))
        print(title)
        print(f"  {'score':18s}" + "".join(f"{k:>10s}" for k in keys))
        print("  " + "-" * (18 + 10 * len(keys)))
        for s in rows:
            print(f"  {s:18s}" + "".join(
                f"{cells[s][k]:10.4f}" if isinstance(cells[s].get(k), (int, float))
                else f"{'-':>10s}" for k in keys))
        print()

    n_ds = len({r["dataset"] for r in dataset_rows})
    table(f"{args.metric} by samples-per-class   [MEAN over {n_ds} dataset(s)]",
          pivot, settings)

    by_ds: dict[tuple[str, str], dict[str, float]] = {}
    for r in dataset_rows:
        by_ds.setdefault((r["dataset"], r["score"]), {})[r["samples_per_class"]] = r.get(args.metric)
    with (out / f"pivot_{args.metric}_by_dataset.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["dataset", "score", *settings])
        for (d, s) in sorted(by_ds):
            w.writerow([d, s, *[by_ds[(d, s)].get(k, "") for k in settings]])

    if args.by_dataset:
        for d in sorted({d for d, _ in by_ds}):
            table(f"{args.metric} - {d}",
                  {s: v for (dd, s), v in by_ds.items() if dd == d}, settings)

    print(f"written to {out}")
    print(f"  summary_by_spc.csv               mean over datasets, every metric")
    print(f"  per_dataset.csv                  one row per (spc, score, dataset)")
    print(f"  pivot_{args.metric}.csv{'':{max(0, 26 - len(args.metric))}s}scores x spc, averaged")
    print(f"  pivot_{args.metric}_by_dataset.csv{'':{max(0, 15 - len(args.metric))}s}scores x spc, per dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
