"""Explicit simulation actions, independent from any policy or device adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from .arena import Point, TICK_MS


class ActionKind(StrEnum):
    WAIT = "WAIT"
    PLAY_CARD = "PLAY_CARD"
    PLAY_SPELL = "PLAY_SPELL"
    END_STEP = "END_STEP"


@dataclass(frozen=True)
class SimAction:
    kind: ActionKind
    timestamp_ms: int
    card: Optional[str] = None
    slot: Optional[int] = None
    position: Optional[Point] = None
    duration_ms: int = TICK_MS

    @classmethod
    def wait(cls, timestamp_ms: int, duration_ms: int = TICK_MS) -> "SimAction":
        return cls(ActionKind.WAIT, timestamp_ms, duration_ms=duration_ms)

    @classmethod
    def play_card(cls, timestamp_ms: int, card: str, position: Point,
                  slot: Optional[int] = None) -> "SimAction":
        return cls(ActionKind.PLAY_CARD, timestamp_ms, card=card, slot=slot,
                   position=position)

    @classmethod
    def play_spell(cls, timestamp_ms: int, card: str, position: Point,
                   slot: Optional[int] = None) -> "SimAction":
        return cls(ActionKind.PLAY_SPELL, timestamp_ms, card=card, slot=slot,
                   position=position)

    @classmethod
    def end_step(cls, timestamp_ms: int, duration_ms: int = TICK_MS) -> "SimAction":
        return cls(ActionKind.END_STEP, timestamp_ms, duration_ms=duration_ms)


def apply_action(match, side: int, action: SimAction) -> bool:
    """Apply one policy action to a Match without involving rendering or ADB."""
    if action.kind in {ActionKind.WAIT, ActionKind.END_STEP}:
        match.step(max(1, int(action.duration_ms)))
        return True
    if action.kind not in {ActionKind.PLAY_CARD, ActionKind.PLAY_SPELL}:
        raise ValueError(f"unsupported action kind: {action.kind}")
    if not action.card or action.position is None:
        return False
    return bool(match.play_card(side, action.card, action.position))
