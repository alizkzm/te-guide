"""SPAN - update span across short adaptation runs, in ITM's shape."""

from __future__ import annotations

import numpy as np

from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import ScoreNotApplicable, TEScore, as_float_features


def span_descriptors(F: np.ndarray) -> dict[str, float]:
    """Geometry of a [K, D] stack of update directions."""
    F = np.asarray(F, dtype=np.float64)
    norms = np.linalg.norm(F, axis=1)
    good = norms > 1e-12
    if int(good.sum()) < 2:
        raise ScoreNotApplicable(
            f"SPAN needs >=2 non-degenerate update directions, got {int(good.sum())}: "
            f"the adaptation did not move the representation. Raise `lr` or `iters`."
        )
    Fn = F[good] / norms[good][:, None]
    C = Fn @ Fn.T
    k = len(C)
    off = C[~np.eye(k, dtype=bool)]
    ev = np.clip(np.linalg.eigvalsh((C + C.T) / 2), 0.0, None)
    p = ev / max(ev.sum(), 1e-12)
    p = p[p > 1e-12]
    ent = float(-(p * np.log(p)).sum())
    return {
        "span_effective_rank": float(np.exp(ent)),
        "span_eff_rank_norm": float(np.exp(ent) / k),
        "span_spectral_entropy": ent,
        "span_isotropy": float(ev.min() / max(ev.max(), 1e-12)),
        "span_concentration": float(ev.max() / max(ev.sum(), 1e-12)),
        "span_mean_abs_cos": float(np.abs(off).mean()),
        "span_mean_cos": float(off.mean()),
        "span_ftv_norm": float(norms[good].mean()),
        "span_ftv_norm_cv": float(norms[good].std() / max(norms[good].mean(), 1e-12)),
        "span_k_used": float(good.sum()),
    }


def _splits(n: int, probe_size: int, train_size: int, n_subsets: int, seed: int):
    """(probe_idx, [train_idx ...]) - probe fixed, every subset disjoint from it."""
    import torch

    probe_size = int(min(probe_size, max(n // 2, 1)))
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    probe = perm[:probe_size].numpy()
    pool = perm[probe_size:]
    if len(pool) < 2:
        raise ScoreNotApplicable(f"probe has only {n} samples - too few for SPAN")
    k_take = int(min(train_size, len(pool)))
    trains = []
    for k in range(int(n_subsets)):
        gk = torch.Generator().manual_seed(int(seed) + 1000 + k)
        trains.append(pool[torch.randperm(len(pool), generator=gk)[:k_take]].numpy())
    return probe, trains


def _build_dva(n_classes, d, m, coeff, model_size, device):
    """ITM's DVA, verbatim from guide/scores/dynamics.py, plus a label-free readout."""
    import torch
    from torch import nn

    class DVA(nn.Module):
        def __init__(self, c, dim, m, coeff, model_size):
            super().__init__()
            layers = []
            for _ in range(model_size):
                layers += [nn.Linear(dim, dim), nn.ReLU()]
            layers.append(nn.Linear(dim, c))
            self.tmodel = nn.Sequential(*layers)
            self.lp = nn.Linear(c, c)
            self.m, self.coeff = m, coeff

        def rep(self, x):
            return torch.nn.functional.normalize(self.tmodel(x), p=2, dim=1)

        def hidden(self, x):
            h = x
            for layer in list(self.tmodel)[:-1]:
                h = layer(h)
            return h

        def forward(self, x, y_embed):
            x = self.rep(x)
            nb = x.shape[0]
            X = x.detach()
            v = x - y_embed
            s = self.coeff / nb
            for _ in range(self.m):
                v = v - s * (X @ (X.t() @ v))
            ym = v + y_embed
            return ym, self.lp(ym)

    return DVA(n_classes, d, m, coeff, model_size).to(device)


@register
class SPAN(TEScore):
    name = "span"
    paper = "SPAN: effective span of cross-run update directions (ITM-style adaptation)"
    elements = (
        "span_effective_rank",
        "span_eff_rank_norm",
        "span_spectral_entropy",
        "span_isotropy",
        "span_concentration",
        "span_mean_abs_cos",
        "span_mean_cos",
        "span_ftv_norm",
        "span_ftv_norm_cv",
        "span_probe_acc",
        "span_k_used",
    )
    requires = ("features", "labels")
    hparams = {
        "n_subsets": 10,
        "train_size": 500,
        "probe_size": 200,
        "iters": 100,
        "readout": "rep",
        "batch_size": 256,
        "lr": 0.005,
        "weight_decay": 0.01,
        "m": 1,
        "coeff": 0.5,
        "model_size": 0,
        "seed": 12357,
        "device": "cpu",
    }
    note = (
        "Same adaptation as `itm` (DVA on cached probe features) run K times on K "
        "subsets; the readout is the geometry of the update directions instead of "
        "the best validation accuracy. `span_ftv_norm` is the control - if it "
        "tracks accuracy as well as `span_effective_rank`, the score reads how far "
        "the module moved, not the geometry of where it moved."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        import torch
        from torch import nn, optim

        X = as_float_features(probe)
        y = np.asarray(probe.labels).astype(np.int64)
        n, d = X.shape
        n_classes = int(y.max()) + 1
        if n_classes < 2:
            raise ScoreNotApplicable("SPAN needs at least 2 classes")

        readout = str(hp["readout"]).lower()
        model_size = int(hp["model_size"])
        if readout == "hidden" and model_size < 1:
            raise ScoreNotApplicable("readout='hidden' needs model_size >= 1")

        device = torch.device(str(hp["device"]))
        probe_idx, trains = _splits(
            n, int(hp["probe_size"]), int(hp["train_size"]),
            int(hp["n_subsets"]), int(hp["seed"]),
        )

        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp_min(1e-6)
        Xt = (Xt - mu) / sd
        yt = torch.as_tensor(y, dtype=torch.long, device=device)
        Xp = Xt[torch.as_tensor(probe_idx, device=device)]
        yp = yt[torch.as_tensor(probe_idx, device=device)]

        proto = nn.Embedding(n_classes, n_classes)
        w = torch.nn.functional.one_hot(torch.arange(n_classes), n_classes).float()
        proto.weight = nn.Parameter(torch.nn.functional.normalize(w, p=2, dim=1))
        proto.requires_grad_(False)
        proto = proto.to(device)

        read = (lambda mdl, x: mdl.rep(x)) if readout == "rep" else (lambda mdl, x: mdl.hidden(x))

        torch.manual_seed(int(hp["seed"]))
        with torch.no_grad():
            v0 = read(_build_dva(n_classes, d, int(hp["m"]), float(hp["coeff"]),
                                 model_size, device), Xp).clone()

        lossf = nn.CrossEntropyLoss()
        directions, accs = [], []

        for k, tr in enumerate(trains):
            torch.manual_seed(int(hp["seed"]))
            model = _build_dva(n_classes, d, int(hp["m"]), float(hp["coeff"]),
                               model_size, device)
            opt = optim.AdamW(model.parameters(), lr=float(hp["lr"]),
                              weight_decay=float(hp["weight_decay"]))
            sch = optim.lr_scheduler.PolynomialLR(opt, total_iters=int(hp["iters"]), power=1.0)

            idx = torch.as_tensor(tr, device=device)
            Xk, yk = Xt[idx], yt[idx]
            bs = int(min(int(hp["batch_size"]), len(idx)))
            g = torch.Generator().manual_seed(int(hp["seed"]) + 1000 + k)
            n_iters = int(hp["iters"])
            batches = torch.stack(
                [torch.randperm(len(idx), generator=g)[:bs] for _ in range(n_iters)]
            ).to(device)

            model.train()
            for step in range(n_iters):
                sel = batches[step]
                xb, yb = Xk[sel], yk[sel]
                _, logits = model(xb, proto(yb))
                opt.zero_grad()
                lossf(logits, yb).backward()
                opt.step()
                sch.step()

            model.eval()
            with torch.no_grad():
                directions.append((read(model, Xp) - v0).mean(0).cpu().numpy())
                pred = model.lp(model.rep(Xp)).argmax(1)
                accs.append(float((pred == yp).float().mean()))

        out = span_descriptors(np.stack(directions))
        out["span_probe_acc"] = float(np.mean(accs))
        return out


def span_artifacts(probe: ProbeData, n_dirs: int = 5, **hp) -> dict:
    """Raw geometry from a SPAN run, for cross-model / cross-task analysis."""
    import torch
    from torch import nn, optim

    full = dict(SPAN.hparams); full.update(hp)
    X = as_float_features(probe)
    y = np.asarray(probe.labels).astype(np.int64)
    n, d = X.shape
    n_classes = int(y.max()) + 1
    if n_classes < 2:
        raise ScoreNotApplicable("SPAN needs at least 2 classes")

    device = torch.device(str(full["device"]))
    probe_idx, trains = _splits(n, int(full["probe_size"]), int(full["train_size"]),
                                int(full["n_subsets"]), int(full["seed"]))
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Xt = (Xt - Xt.mean(0, keepdim=True)) / Xt.std(0, keepdim=True).clamp_min(1e-6)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    Xp = Xt[torch.as_tensor(probe_idx, device=device)]

    proto = nn.Embedding(n_classes, n_classes)
    w = torch.nn.functional.one_hot(torch.arange(n_classes), n_classes).float()
    proto.weight = nn.Parameter(torch.nn.functional.normalize(w, p=2, dim=1))
    proto.requires_grad_(False)
    proto = proto.to(device)

    torch.manual_seed(int(full["seed"]))
    with torch.no_grad():
        v0 = _build_dva(n_classes, d, int(full["m"]), float(full["coeff"]),
                        int(full["model_size"]), device).rep(Xp).clone()

    lossf = nn.CrossEntropyLoss()
    ftvs, shifts, mats = [], [], []
    for k, tr in enumerate(trains):
        torch.manual_seed(int(full["seed"]))
        model = _build_dva(n_classes, d, int(full["m"]), float(full["coeff"]),
                           int(full["model_size"]), device)
        opt = optim.AdamW(model.parameters(), lr=float(full["lr"]),
                          weight_decay=float(full["weight_decay"]))
        idx = torch.as_tensor(tr, device=device)
        Xk, yk = Xt[idx], yt[idx]
        bs = int(min(int(full["batch_size"]), len(idx)))
        g = torch.Generator().manual_seed(int(full["seed"]) + 1000 + k)
        batches = torch.stack([torch.randperm(len(idx), generator=g)[:bs]
                               for _ in range(int(full["iters"]))]).to(device)
        model.train()
        for step in range(int(full["iters"])):
            sel = batches[step]
            _, logits = model(Xk[sel], proto(yk[sel]))
            opt.zero_grad()
            lossf(logits, yk[sel]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            delta = (model.rep(Xp) - v0)
            shifts.append(delta.cpu().numpy())
            ftvs.append(delta.mean(0).cpu().numpy())
            mats.append(list(model.tmodel)[-1].weight.detach().cpu().numpy())

    W = np.mean(np.stack(mats), axis=0)
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    r = int(max(1, min(n_dirs, Vt.shape[0])))
    return dict(ftv=np.stack(ftvs), shift=np.mean(np.stack(shifts), axis=0),
                subspace=Vt[:r].T, n_classes=n_classes, feature_dim=d)


def subspace_affinity(A: np.ndarray, B: np.ndarray) -> float:
    """||A^T B||_F^2 / r for orthonormal bases - 1 = identical span, 0 = orthogonal."""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    if A.shape[0] != B.shape[0]:
        return float("nan")
    r = min(A.shape[1], B.shape[1])
    return float((np.linalg.norm(A[:, :r].T @ B[:, :r], ord="fro") ** 2) / r)


@register
class SPANRandomTask(SPAN):
    """Control: the same machinery with the labels shuffled."""

    name = "span_random_task"
    paper = "SPAN control: update span under shuffled labels"
    note = (
        "Control for `span`: same machinery, labels shuffled. Expect it to score "
        "HIGHER than `span` - unlearnable tasks scatter the update directions. It "
        "bounds the noise ceiling, not a floor."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        y = np.asarray(probe.labels)
        shuffled = np.random.RandomState(int(hp["seed"])).permutation(y)
        fake = ProbeData(
            model_name=probe.model_name, dataset=probe.dataset,
            source_data=probe.source_data, features=probe.features,
            labels=shuffled.astype(np.int64), logits=probe.logits,
            weights=probe.weights, meta=probe.meta,
        )
        return SPAN.compute(self, fake, **hp)
