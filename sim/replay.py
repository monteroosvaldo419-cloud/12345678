"""Minimal deterministic replay records for offline inspection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .actions import ActionKind, SimAction
from .arena import Point


@dataclass(frozen=True)
class ReplayRecord:
    timestamp_ms: int
    kind: str
    card: Optional[str] = None
    slot: Optional[int] = None
    x: Optional[int] = None
    y: Optional[int] = None
    duration_ms: int = 0

    @classmethod
    def from_action(cls, action: SimAction) -> "ReplayRecord":
        position = action.position
        return cls(
            timestamp_ms=action.timestamp_ms,
            kind=str(action.kind),
            card=action.card,
            slot=action.slot,
            x=None if position is None else position.x,
            y=None if position is None else position.y,
            duration_ms=action.duration_ms,
        )

    def to_action(self) -> SimAction:
        position = None if self.x is None or self.y is None else Point(self.x, self.y)
        return SimAction(
            kind=ActionKind(self.kind),
            timestamp_ms=self.timestamp_ms,
            card=self.card,
            slot=self.slot,
            position=position,
            duration_ms=self.duration_ms,
        )


@dataclass
class Replay:
    seed: int
    actions: list[ReplayRecord]
    events: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "seed": self.seed,
            "actions": [asdict(action) for action in self.actions],
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Replay":
        return cls(
            seed=int(payload.get("seed", 0)),
            actions=[ReplayRecord(**item) for item in payload.get("actions", [])],
            events=list(payload.get("events", [])),
        )

    def write(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def read(cls, path: Path) -> "Replay":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def make_replay(seed: int, actions: Iterable[SimAction], diagnostics=None) -> Replay:
    events = [] if diagnostics is None else list(diagnostics.events)
    return Replay(seed=seed,
                  actions=[ReplayRecord.from_action(action) for action in actions],
                  events=events)
