"""The shared state every agent reads and writes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Workspace:
    topics: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    plan: list = field(default_factory=list)       # [{date, topic, hours, kind}]
    scores: dict = field(default_factory=dict)     # topic -> [0..1, ...]
    attempts: list = field(default_factory=list)   # quiz history
    exam_date: str | None = None
    hours_per_day: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Workspace":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})
