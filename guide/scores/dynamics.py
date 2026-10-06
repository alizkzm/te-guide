"""TE scores that *simulate* part of fine-tuning: PED, LEAD, ITM."""

from __future__ import annotations

import numpy as np

from guide.core.linalg import one_hot, softmax
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import TEScore, as_float_features
from guide.scores.classic import logme_score
from guide.scores.separability import SFDA


def _per_class_mean_var(X: np.ndarray, y: np.ndarray):
    classes = np.unique(y)
    means = np.stack([X[y == c].mean(axis=0) for c in classes])
    varis = np.stack([X[y == c].var(axis=0) for c in classes])
    return means, varis


def one_step_ped(
    X: np.ndarray,
    y: np.ndarray,
    general_X: np.ndarray | None = None,
    time_step: float = 0.1,
    eps: float = 1e-3,
    nsigma: float = 0.6,
    constant: float = 1.0,
):
    """One Coulomb-style repulsion step between overlapping class clusters."""
    ref = general_X if general_X is not None else X
    smean = ref.mean(axis=0)
    sstd = np.maximum(ref.std(axis=0), eps)
    Xn = (X - smean) / sstd

    means, varis = _per_class_mean_var(Xn, y)
    n_classes, d = means.shape
    radius = np.sqrt(varis.sum(axis=1))

    gi, gj = np.meshgrid(np.arange(n_classes), np.arange(n_classes))
    mask = np.triu(gi != gj, k=1)
    ii, jj = gi[mask], gj[mask]

    dist = means[ii] - means[jj]
    dnorm = np.sqrt((dist ** 2).sum(axis=1))
    overlap = np.maximum(abs(nsigma) * radius[ii] + abs(nsigma) * radius[jj] - dnorm, 0.0)
    direction = dist / (dnorm.reshape(-1, 1) + 1e-12)
    accel = constant * overlap.reshape(-1, 1)

    shift = np.zeros((n_classes, n_classes, d))
    shift[mask] = -direction * np.minimum(
        0.5 * time_step ** 2 * accel, overlap.reshape(-1, 1)
    )
    shift = shift - np.transpose(shift, (1, 0, 2))
    total = shift.sum(axis=1)

    Xn = Xn + total[y]
    Xout = Xn * sstd + smean
    return Xout, float(np.abs(total).sum())


def ped_remap(
    X: np.ndarray,
    y: np.ndarray,
    general_X: np.ndarray | None = None,
    exit_ratio: float = 0.5,
    time_step: float = 0.1,
    nsigma: float = 0.6,
    max_steps: int = 6,
) -> np.ndarray:
    X = X.copy()
    first_change = None
    for step in range(max_steps):
        X_new, change = one_step_ped(
            X, y, general_X=general_X, time_step=time_step, nsigma=nsigma
        )
        if step == 0:
            first_change = change if change > 0 else 1.0
        if change / first_change > exit_ratio:
            X = X_new
        else:
            break
    return X


@register
class PED(TEScore):
    name = "ped"
    paper = "Exploring Model Transferability through the Lens of Potential Energy (ICCV 2023)"
    elements = ("ped_sfda", "ped_logme")
    hparams = {
        "exit_ratio": 0.5,
        "time_step": 0.1,
        "nsigma": 0.6,
        "max_steps": 6,
        "use_source_stats": True,
        "base_metrics": ("sfda", "logme"),
    }
    note = (
        "PED is a feature *remapping*; the score is whatever base metric you run "
        "on the remapped features. Rescaling by ImageNet feature statistics "
        "(paper App. B) uses the source probe when it is available."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels

        general_X = None
        if hp.get("use_source_stats"):
            src = probe.attach_source_probe(required=False)
            if src is not None and src.features is not None:
                general_X = np.asarray(src.features, dtype=np.float64)

        Xr = ped_remap(
            X,
            y,
            general_X=general_X,
            exit_ratio=float(hp["exit_ratio"]),
            time_step=float(hp["time_step"]),
            nsigma=float(hp["nsigma"]),
            max_steps=int(hp["max_steps"]),
        )

        out: dict[str, float] = {}
        bases = tuple(hp["base_metrics"])
        if "sfda" in bases:
            remapped = ProbeData(
                model_name=probe.model_name,
                dataset=probe.dataset,
                features=Xr,
                labels=y,
            )
            out["ped_sfda"] = float(SFDA().value(remapped))
        if "logme" in bases:
            out["ped_logme"] = logme_score(Xr, y)
        return out


@register
class LEAD(TEScore):
    name = "lead"
    paper = "LEAD: Exploring Logit Space Evolution for Model Selection (CVPR 2024)"
    elements = ("lead",)
    hparams = {"S": 8, "t": 10.0, "H": 64, "ridge": 1e-3, "init": "lstsq", "seed": 0}
    status = "ready"
    note = (
        "Faithful re-implementation of Algorithm 1 (no official code). Init logits "
        "from a ridge classifier C; a randomly-initialised 1-hidden-layer MLP head "
        "gives the NTK Phi per class; mean eigenvalue lambda_bar sets a closed-form "
        "exponential relaxation of logits toward the labels; score = -cross-entropy "
        "of the predicted final logits. Constant-NTK (Phi frozen at head init), as "
        "in the paper."
    )

    def compute(self, probe, **hp):
        rng = np.random.default_rng(int(hp["seed"]))
        Z, y = as_float_features(probe), probe.labels
        n, d = Z.shape; K = int(y.max() + 1); Y = one_hot(y, K)
        Z = Z - Z.mean(0, keepdims=True)
        Z = Z / (np.linalg.norm(Z, axis=1).mean() + 1e-12)
    
        W = np.linalg.solve(Z.T @ Z + float(hp["ridge"]) * np.eye(d), Z.T @ Y)
        log_init = Z @ W
    
        H = int(hp["H"]); s = 1.0 / np.sqrt(d)
        W1 = rng.standard_normal((d, H)) * s;  W2 = rng.standard_normal((H, K)) / np.sqrt(H)
    
        def jac_sum(z):
            a = z @ W1; h = np.tanh(a); dh = 1 - h**2
            w2sum = W2.sum(1)
            gW1 = np.outer(z, w2sum * dh).ravel()
            gb1 = w2sum * dh
            gW2 = np.tile(h, K); gb2 = np.ones(K)
            return np.concatenate([gW1, gb1, gW2, gb2])
    
        S = int(hp["S"]); t = float(hp["t"])
        F_pred = log_init.copy(); lam_list = []
        for k in range(K):
            idx = np.where(y == k)[0]
            if idx.size == 0: continue
            sel = idx if idx.size <= S else rng.choice(idx, S, replace=False)
            G = np.stack([jac_sum(Z[i]) for i in sel])
            Phi = G @ G.T
            lam = np.linalg.eigvalsh(Phi)
            lam_bar = lam.mean() / (lam.sum() + 1e-12)
            lam_list.append(lam_bar * t)
            decay = np.exp(-lam_bar * t)
            F_pred[idx] = (1 - decay) * Y[idx] + decay * log_init[idx]
    
        ce = -np.log(softmax(F_pred)[np.arange(n), y] + 1e-12).mean()
        self._last_lam_t = (float(np.min(lam_list)), float(np.max(lam_list)))
        return {"lead": float(-ce)}


@register
class ITM(TEScore):
    name = "itm"
    paper = "Implicit Modeling for Transferability Estimation of Vision Foundation Models (2025)"
    elements = ("itm",)
    hparams = {
        "batch_size": 256,
        "lr": 0.005,
        "weight_decay": 0.01,
        "max_iter": 10000,
        "train_iter": 500,
        "m": 1,
        "coeff": 0.5,
        "val_interval": 100,
        "model_size": 0,
        "seed": 12357,
        "device": "cpu",
    }
    note = (
        "Port of BUAAHugeGun/ITM (TE/itm.py + nets/dva.py + nets/mk_proto.py). "
        "This is the only score that trains a small network, so it is by far the "
        "slowest (~500 AdamW iterations per model/dataset pair)."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader, TensorDataset, random_split

        torch.manual_seed(int(hp["seed"]))
        np.random.seed(int(hp["seed"]))

        device = torch.device(hp["device"])
        f = torch.tensor(as_float_features(probe), dtype=torch.float32)
        mean, std = f.mean(dim=0), f.std(dim=0)
        f = (f - mean) / torch.clamp(std, min=1e-6)
        y = torch.tensor(probe.labels, dtype=torch.long)

        n_classes = int(y.max().item() + 1)
        d = f.shape[1]

        proto = nn.Embedding(n_classes, n_classes)
        w = torch.nn.functional.one_hot(torch.arange(n_classes), n_classes).float()
        proto.weight = nn.Parameter(torch.nn.functional.normalize(w, p=2, dim=1))
        proto.requires_grad_(False)
        proto = proto.to(device)

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

            def forward(self, x, y_embed):
                x = self.tmodel(x)
                x = torch.nn.functional.normalize(x, p=2, dim=1)
                nb = x.shape[0]
                X = x.detach()
                B = (X @ X.t()) * self.coeff / nb
                A = torch.eye(nb, device=x.device) - B
                ym = torch.linalg.matrix_power(A, self.m) @ (x - y_embed) + y_embed
                return ym, self.lp(ym)

        model = DVA(n_classes, d, int(hp["m"]), float(hp["coeff"]), int(hp["model_size"])).to(device)

        ds = TensorDataset(f, y)
        n_train = int(0.8 * len(ds))
        train_ds, val_ds = random_split(
            ds, [n_train, len(ds) - n_train], generator=torch.Generator().manual_seed(int(hp["seed"]))
        )
        bs = int(hp["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        opt = optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
        sch = optim.lr_scheduler.PolynomialLR(opt, total_iters=int(hp["max_iter"]), power=1.0)

        accs: list[float] = []
        it = iter(train_loader)
        for i in range(int(hp["train_iter"])):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(train_loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            _, logits = model(xb, proto(yb))
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()

            if (i + 1) % int(hp["val_interval"]) == 0:
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        _, logits = model(xb, proto(yb))
                        correct += int((logits.argmax(1) == yb).sum())
                        total += int(yb.numel())
                accs.append(correct / max(total, 1))
                model.train()

        return {"itm": float(max(accs)) if accs else float("nan")}
