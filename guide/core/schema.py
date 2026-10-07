"""The on-disk score record."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from guide import __version__, config


@dataclass
class ScoreRecord:
    target_data: str
    model_name: str
    score_name: str
    source_data: str = config.SOURCE_DATA

    samples_per_class: str | int = "all"
    run_time: float = 0.0
    score_elements_list: dict[str, float] = field(default_factory=dict)
    score_element_names: list[str] = field(default_factory=list)

    score_kind: str = "TE"
    score_completeness: str = "complete"
    score_paper: str = ""
    score_note: str = ""
    label_free: bool = False

    n_samples: int = 0
    n_classes: int = 0
    feature_dim: int = 0
    hyperparameters: dict = field(default_factory=dict)
    seed: int = config.SEED
    status: str = "ok"
    error: str | None = None
    created_at: str = ""
    guide_version: str = __version__

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.score_element_names:
            self.score_element_names = list(self.score_elements_list)

    @property
    def filename(self) -> str:
        return f"{self.target_data}_{self.model_name}_{self.score_name}.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score_elements_list"] = {
            k: (None if v is None or not math.isfinite(float(v)) else float(v))
            for k, v in self.score_elements_list.items()
        }
        return d


def record_path(
    target_data: str,
    model_name: str,
    score_name: str,
    source_data: str = config.SOURCE_DATA,
    root: Path | None = None,
) -> Path:
    root = root or config.SCORE_DIR
    return Path(root) / source_data / f"{target_data}_{model_name}_{score_name}.json"


def save_record(record: ScoreRecord, root: Path | None = None) -> Path:
    path = record_path(
        record.target_data,
        record.model_name,
        record.score_name,
        record.source_data,
        root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def load_record(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_records(
    source_data: str = config.SOURCE_DATA,
    target_data: str | None = None,
    score_name: str | None = None,
    root: Path | None = None,
):
    """Yield every stored record dict, optionally filtered."""
    root = Path(root or config.SCORE_DIR) / source_data
    if not root.exists():
        return
    for path in sorted(root.glob("*.json")):
        rec = load_record(path)
        if target_data is not None and rec.get("target_data") != target_data:
            continue
        if score_name is not None and rec.get("score_name") != score_name:
            continue
        yield rec
