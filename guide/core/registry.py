"""Score registry: `@register` a `TEScore` subclass, get it back by name."""

from __future__ import annotations

REGISTRY: dict[str, type] = {}


def register(cls=None, /, **overrides):
    """Class decorator."""

    def _wrap(klass):
        for k, v in overrides.items():
            setattr(klass, k, v)
        name = getattr(klass, "name", None)
        if not name:
            raise ValueError(f"{klass.__name__} must define a `name`")
        if name in REGISTRY and REGISTRY[name] is not klass:
            raise ValueError(f"duplicate score name: {name!r}")
        REGISTRY[name] = klass
        return klass

    return _wrap if cls is None else _wrap(cls)


def get_score(name: str):
    _ensure_loaded()
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(
            f"unknown score {name!r}. available: {', '.join(sorted(REGISTRY))}"
        ) from None


def list_scores(kind: str | None = None, include_disabled: bool = False) -> list[str]:
    """`kind` in {None, 'te', 'trivial', 'label_free', 'ready', 'partial'}."""
    from guide import config

    _ensure_loaded()
    names = sorted(REGISTRY)
    if not include_disabled:
        names = [n for n in names if n not in config.DISABLED_SCORES]
    if kind in (None, "all"):
        return names
    out = []
    for n in names:
        cls = REGISTRY[n]
        if kind == "te" and not getattr(cls, "trivial", False):
            out.append(n)
        elif kind == "trivial" and getattr(cls, "trivial", False):
            out.append(n)
        elif kind == "label_free" and getattr(cls, "label_free", False):
            out.append(n)
        elif kind == "ready" and getattr(cls, "status", "ready") == "ready":
            out.append(n)
        elif kind == "partial" and getattr(cls, "status", "ready") != "ready":
            out.append(n)
    return out


_LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if not _LOADED:
        _LOADED = True
        import guide.scores
