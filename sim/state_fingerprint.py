"""Compact observable state fingerprints for replay and regression checks."""

from __future__ import annotations

import hashlib
import json


def state_fingerprint(match) -> str:
    entities = []
    for entity in sorted(match.battle.entities.values(), key=lambda item: item.uid):
        entities.append((
            entity.uid, entity.name, entity.side, entity.alive,
            entity.pos.x, entity.pos.y, entity.hitpoints, entity.target_uid,
            entity.attack_cooldown_ms, entity.deploy_remaining_ms,
        ))
    players = []
    for side in (1, -1):
        player = match.players[side]
        players.append((side, player.elixir, tuple(player.hand), tuple(player.queue)))
    payload = (match.elapsed_ms, match.finished, match.result,
               tuple(players), tuple(entities))
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def same_state(left, right) -> bool:
    return state_fingerprint(left) == state_fingerprint(right)