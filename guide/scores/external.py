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
