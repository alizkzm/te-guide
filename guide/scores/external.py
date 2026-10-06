"""Scores that need an artefact GUIDE's probe cache does not contain."""

from __future__ import annotations

import numpy as np

from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import ScoreNotApplicable, SingleElementScore, TEScore


def task2vec_embedding(model, dataloader, criterion, n_batches: int = 100, device="cpu"):
    """Diagonal Fisher information of a frozen probe network = the task embedding."""
    import torch

    model = model.to(device).eval()
    fisher = None
    seen = 0
    for i, (x, target) in enumerate(dataloader):
        if i >= n_batches:
            break
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x.to(device)), target.to(device))
        loss.backward()
        grads = [
            (p.grad.detach() ** 2).cpu() for p in model.parameters() if p.grad is not None
        ]
        fisher = grads if fisher is None else [f + g for f, g in zip(fisher, grads)]
        seen += 1
    if fisher is None:
        raise RuntimeError("no gradients collected")
    flat = torch.cat([f.flatten() for f in fisher]) / max(seen, 1)
    return flat.numpy()


def task2vec_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric cosine distance on sqrt-Fisher vectors (lower = closer tasks)."""
    a, b = np.sqrt(a), np.sqrt(b)
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@register
class Task2Vec(SingleElementScore):
    name = "task2vec"
    paper = "Task2Vec: Task Embedding for Meta-Learning (ICCV 2019)"
    elements = ("task2vec",)
    requires = ()
    status = "partial"
    note = (
        "Needs per-sample BACKWARD passes through a probe network, not cached "
        "features. Use `guide.scores.external.task2vec_embedding(model, loader, "
        "criterion)` on the source and target tasks and score with "
        "`-task2vec_distance(e_src, e_tgt)`. Not wired into the probe pipeline "
        "because it is a task-to-task, not a model-to-task, quantity."
    )

    def value(self, probe: ProbeData, **hp) -> float:
        raise ScoreNotApplicable(self.note)


def principal_gradient_expectation(model, loader, criterion, top_k=10, device="cpu"):
    import torch

    model = model.to(device).eval()
    grads = []
    for x, y in loader:
        model.zero_grad(set_to_none=True)
        criterion(model(x.to(device)), y.to(device)).backward()
        g = torch.cat(
            [p.grad.detach().flatten().cpu() for p in model.parameters() if p.grad is not None]
        )
        grads.append(g.numpy())
    G = np.stack(grads)
    _, _, vh = np.linalg.svd(G - G.mean(0, keepdims=True), full_matrices=False)
    return vh[:top_k].mean(axis=0)


@register
class PGE(SingleElementScore):
    name = "pge"
    paper = "Transferability Estimation Based On Principal Gradient Expectation (2022)"
    elements = ("pge",)
    requires = ()
    status = "partial"
    note = (
        "Needs gradients on both source and target data. Helper provided: "
        "`principal_gradient_expectation(model, loader, criterion)`; score = "
        "cosine(g_source, g_target). No public reference implementation exists."
    )

    def value(self, probe: ProbeData, **hp) -> float:
        raise ScoreNotApplicable(self.note)


def emms_score(X: np.ndarray, label_embeddings: list[np.ndarray], n_iters: int = 50) -> float:
    """Weighted linear-square regression from features onto a mixture of foundation-model label embeddings (alternating minimisation)."""
    k = len(label_embeddings)
    lam = np.ones(k) / k
    W = None
    for _ in range(n_iters):
        Y = sum(l * e for l, e in zip(lam, label_embeddings))
        W = np.linalg.lstsq(X, Y, rcond=None)[0]
        pred = X @ W
        res = np.array([np.linalg.norm(pred - e) ** 2 for e in label_embeddings])
        lam = 1.0 / (res + 1e-8)
        lam /= lam.sum()
    Y = sum(l * e for l, e in zip(lam, label_embeddings))
    return float(-np.mean((X @ W - Y) ** 2))


@register
class EMMS(SingleElementScore):
    name = "emms"
    paper = "Foundation Model is Efficient Multimodal Multitask Model Selector (NeurIPS 2023)"
    elements = ("emms",)
    requires = ("features", "labels")
    status = "partial"
    hparams = {"label_embedding_dir": None}
    note = (
        "Needs label embeddings produced by foundation models (CLIP / GPT-2 / "
        "BERT text encoders) for the target label set. Compute them once with "
        "OpenGVLab/Multitask-Model-Selector, drop the .npy files in "
        "`label_embedding_dir`, and `emms_score` will run. GUIDE does not ship "
        "those encoders."
    )

    def value(self, probe: ProbeData, **hp) -> float:
        from pathlib import Path

        d = hp.get("label_embedding_dir")
        if not d:
            raise ScoreNotApplicable(self.note)
        paths = sorted(Path(d).glob(f"{probe.dataset}_*.npy"))
        if not paths:
            raise ScoreNotApplicable(f"no label embeddings for {probe.dataset} in {d}")
        embeddings = [np.load(p).astype(np.float64) for p in paths]
        return emms_score(np.asarray(probe.features, dtype=np.float64), embeddings)


@register
class ModelSpider(TEScore):
    name = "model_spider"
    paper = "Model Spider: Learning to Rank Pre-Trained Models Efficiently (NeurIPS 2023)"
    elements = ("model_spider",)
    requires = ()
    status = "partial"
    hparams = {"model_token_path": None, "task_token_path": None}
    note = (
        "Model-Spider is *learned*: model and task tokens are trained across a "
        "meta-training set of (model, task) pairs, then transferability is their "
        "cosine similarity. GUIDE cannot produce those tokens - train them with "
        "zhangyikaii/Model-Spider and point `model_token_path` / `task_token_path` "
        "at the results. Its per-token features are exactly the other scores in "
        "this package (LEEP/LogME/GBC/NCE/...), which GUIDE does compute."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        mp, tp = hp.get("model_token_path"), hp.get("task_token_path")
        if not mp or not tp:
            raise ScoreNotApplicable(self.note)
        m = np.load(mp).astype(np.float64).ravel()
        t = np.load(tp).astype(np.float64).ravel()
        cos = float(m @ t / (np.linalg.norm(m) * np.linalg.norm(t) + 1e-12))
        return {"model_spider": cos}
