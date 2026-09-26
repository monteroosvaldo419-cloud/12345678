"""Teach the network to imitate the hand-written brain, before asking it to improve.

PPO from a random start does not work on this problem, and the failure is
instructive rather than mysterious. Over 1.25M steps the agent reached
`crowns_for = 0` in *every* evaluation while winning a third of its matches: it
had found the local optimum of spending cheap cards on defence and winning
time-out tiebreaks on tower health. Scoring a crown needs a coordinated sequence
- get elixir, put a tank at the bridge, put the Hog behind it, defend the
counter-push - and the chance of stumbling onto that by sampling from ~2,300
masked actions is negligible. So the reward is dense enough to learn defence and
far too sparse to learn offence.

Behaviour cloning removes the exploration problem instead of tuning around it.
The hand-written policy already knows how to attack; supervised learning on its
decisions gets the network to the same place in minutes, and PPO can then start
from a policy that at least sends a Hog.

Two details that matter for correctness:

**The teacher plays through the environment, not beside it.** Its chosen action
is the action the environment executes, so the recorded states are the states
that actually follow - on-policy for the teacher. Recording a teacher's opinion
of states produced by someone else's actions is a different and much weaker
dataset.

**"Hold" is most of the data.** The brain decides on roughly one step in ten, and
a classifier trained on that imbalance learns to output nothing. Hold examples
are therefore subsampled to a target share.

    python -m sim.clone --episodes 400
    python -m sim.train_ppo --resume tmp/rl/clone.pt --name ppo_from_clone
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sim.env import ClashEnv

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

OUT = ROOT / "tmp" / "rl"
LOG = ROOT / "tmp" / "live" / "rl_train.log"
DIAGNOSTIC_BATCH_VERSION = 1


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} clone: {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def process_rss_mb() -> Optional[float]:
    """Return process RSS when psutil is available for diagnostics."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def save_diagnostic_batch(path: Path, data: dict, actions: np.ndarray,
                         train_idx: np.ndarray | None = None,
                         val_idx: np.ndarray | None = None,
                         metadata: Optional[dict] = None) -> None:
    """Persist one collected clone dataset without changing training data."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"diagnostic batch already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
    meta.update({
        "format_version": DIAGNOSTIC_BATCH_VERSION,
        "num_samples": int(len(actions)),
        "planes_shape": list(data["planes"].shape),
        "scalars_shape": list(data["scalars"].shape),
        "action_mask_shape": list(data["masks"].shape),
        "actions_dtype": str(np.asarray(actions).dtype),
        "episode_count": int(np.unique(data.get("episode_id", [])).size),
    })
    np.savez_compressed(
        path,
        planes=np.asarray(data["planes"]),
        scalars=np.asarray(data["scalars"]),
        action_mask=np.asarray(data["masks"]),
        actions=np.asarray(actions, dtype=np.int64),
        sustainable=np.asarray(data.get("sustainable", np.zeros(len(actions), dtype=bool)), dtype=bool),
        episode_id=np.asarray(data.get("episode_id", np.full(len(actions), -1)), dtype=np.int64),
        sample_index=np.asarray(data.get("sample_index", np.arange(len(actions))), dtype=np.int64),
        train_idx=np.asarray(train_idx if train_idx is not None else [], dtype=np.int64),
        val_idx=np.asarray(val_idx if val_idx is not None else [], dtype=np.int64),
        metadata=np.asarray(json.dumps(meta, sort_keys=True), dtype=np.str_),
    )


def load_diagnostic_batch(path: Path) -> tuple[dict, np.ndarray, dict]:
    """Load a batch saved by :func:`save_diagnostic_batch`."""
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format_version") != DIAGNOSTIC_BATCH_VERSION:
            raise ValueError(f"unsupported diagnostic batch version: {metadata.get('format_version')}")
        data = {
            "planes": archive["planes"].copy(),
            "scalars": archive["scalars"].copy(),
            "masks": archive["action_mask"].copy(),
            "sustainable": archive["sustainable"].copy(),
            "episode_id": (archive["episode_id"].copy()
                           if "episode_id" in archive.files
                           else np.full(len(archive["actions"]), -1, dtype=np.int64)),
            "sample_index": (archive["sample_index"].copy()
                             if "sample_index" in archive.files
                             else np.arange(len(archive["actions"]), dtype=np.int64)),
            "train_idx": archive["train_idx"].copy(),
            "val_idx": archive["val_idx"].copy(),
        }
        actions = archive["actions"].astype(np.int64, copy=True)
    if len(actions) != len(data["planes"]):
        raise ValueError("diagnostic batch arrays have inconsistent lengths")
    return data, actions, metadata


def load_or_collect_batch(load_path: Optional[Path] = None, **collect_kwargs):
    """Load a persisted batch, or preserve the normal collection path."""
    if load_path is not None:
        return load_diagnostic_batch(load_path)
    return collect(**collect_kwargs)


def estimate_hold_keep_rate(raw_hold_rate: float, target_hold_rate: float) -> float:
    """Return the HOLD retention rate needed to reach a target final HOLD share.

    For a teacher raw HOLD rate r and a desired final HOLD share t, the sampled
    dataset satisfies: t = (r * k) / (r * k + (1 - r)), where k is the fraction of
    HOLD labels retained after the subsampling step. Solving for k gives the
    necessary retention probability.
    """
    if not 0.0 < raw_hold_rate < 1.0:
        raise ValueError(f"raw_hold_rate must be in (0, 1): {raw_hold_rate!r}")
    if not 0.0 < target_hold_rate < 1.0:
        raise ValueError(f"target_hold_rate must be in (0, 1): {target_hold_rate!r}")
    keep = target_hold_rate * (1.0 - raw_hold_rate) / (raw_hold_rate * (1.0 - target_hold_rate))
    return float(np.clip(keep, 0.0, 1.0))


def compute_validation_metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    """Metrics for the hierarchical HOLD/NON-HOLD validation contract."""
    from sim.env import CARD_ACTIONS

    target = np.asarray(target, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if target.size == 0:
        return {
            "always_hold_accuracy": 0.0,
            "clone_accuracy": 0.0,
            "hold_accuracy": 0.0,
            "nonhold_accuracy": 0.0,
            "play_accuracy": 0.0,
            "ability_accuracy": 0.0,
            "macro_accuracy": 0.0,
            "gate_accuracy": 0.0,
            "target_hold_share": 0.0,
            "predicted_hold_share": 0.0,
            "target_nonhold_share": 0.0,
            "predicted_nonhold_share": 0.0,
            "validation_wait_rate": 0.0,
            "validation_nonwait_rate": 0.0,
            "nonhold_classes_predicted": 0,
        }

    hold_mask = target == 0
    nonhold_mask = target != 0
    play_mask = (target > 0) & (target < CARD_ACTIONS)
    ability_mask = target >= CARD_ACTIONS
    predicted_hold = predicted == 0
    predicted_nonhold = predicted != 0
    always_hold = np.zeros_like(target)
    always_hold_accuracy = float((always_hold == target).mean())
    clone_accuracy = float((predicted == target).mean())
    hold_accuracy = float((predicted[hold_mask] == target[hold_mask]).mean()) if hold_mask.any() else 0.0
    nonhold_accuracy = float((predicted[nonhold_mask] == target[nonhold_mask]).mean()) if nonhold_mask.any() else 0.0
    play_accuracy = float((predicted[play_mask] == target[play_mask]).mean()) if play_mask.any() else 0.0
    ability_accuracy = float((predicted[ability_mask] == target[ability_mask]).mean()) if ability_mask.any() else 0.0
    macro_accuracy = float((hold_accuracy + nonhold_accuracy) / 2.0) if hold_mask.any() and nonhold_mask.any() else clone_accuracy
    gate_accuracy = float((predicted_hold == hold_mask).mean())
    validation_wait_rate = float((predicted == 0).mean())
    validation_nonwait_rate = float((predicted != 0).mean())
    nonhold_classes_predicted = int(np.unique(predicted[predicted_nonhold]).size) if np.any(predicted_nonhold) else 0
    return {
        "always_hold_accuracy": always_hold_accuracy,
        "clone_accuracy": clone_accuracy,
        "hold_accuracy": hold_accuracy,
        "nonhold_accuracy": nonhold_accuracy,
        "play_accuracy": play_accuracy,
        "ability_accuracy": ability_accuracy,
        "macro_accuracy": macro_accuracy,
        "gate_accuracy": gate_accuracy,
        "target_hold_share": float(hold_mask.mean()),
        "predicted_hold_share": float(predicted_hold.mean()),
        "target_nonhold_share": float(nonhold_mask.mean()),
        "predicted_nonhold_share": float(predicted_nonhold.mean()),
        "validation_wait_rate": validation_wait_rate,
        "validation_nonwait_rate": validation_nonwait_rate,
        "nonhold_classes_predicted": nonhold_classes_predicted,
    }


def checkpoint_selection_key(metrics: dict) -> tuple[float, float, float]:
    """Rank epochs by joint gate/NON-HOLD quality, then action accuracy."""
    gate_gain = metrics["gate_accuracy"] - metrics["always_hold_accuracy"]
    return (
        float(min(gate_gain, metrics["nonhold_accuracy"])),
        float(metrics["play_accuracy"]),
        float(metrics["ability_accuracy"]),
    )


def checkpoint_metadata(epoch: int, metrics: dict,
                        selection_key: tuple[float, ...]) -> dict:
    """Build self-contained metadata for the epoch snapshot being saved."""
    return {
        "epoch": int(epoch),
        "best_epoch": int(epoch),
        "selection_key": tuple(float(value) for value in selection_key),
        "validation_metrics": dict(metrics),
    }


def hierarchical_clone_loss(logits, masks, target):
    """Train the existing heads as HOLD gate plus conditional action policy."""
    if logits.dim() != 2 or masks.dim() != 2:
        raise ValueError("logits and masks must be 2-D tensors")
    if logits.shape != masks.shape:
        raise ValueError("logits and masks must have the same shape")
    if target.dim() != 1 or target.shape[0] != logits.shape[0]:
        raise ValueError("target must have one action per row")
    masks = masks.to(dtype=torch.bool)
    if not bool(masks[:, 0].all()):
        raise ValueError("HOLD action 0 must be legal for every clone sample")
    if target.numel() and (int(target.min()) < 0 or int(target.max()) >= logits.shape[1]):
        raise ValueError("clone target is outside the action space")
    target_legal = masks.gather(1, target.unsqueeze(1)).squeeze(1)
    if not bool(target_legal.all()):
        raise ValueError("clone target is illegal under its action mask")

    target_hold = (target == 0).to(dtype=logits.dtype)
    gate_loss = F.binary_cross_entropy_with_logits(logits[:, 0], target_hold)
    nonhold = target != 0
    if bool(nonhold.any()):
        action_logits = logits[:, 1:].masked_fill(~masks[:, 1:], float("-inf"))
        action_target = target[nonhold] - 1
        action_loss = F.cross_entropy(action_logits[nonhold], action_target)
    else:
        action_loss = logits.sum() * 0.0
    return {"loss": gate_loss + action_loss, "gate_loss": gate_loss, "action_loss": action_loss}


def summarize_action_distribution(actions: np.ndarray | list[int]) -> dict:
    """Compact stats for BEFORE/AFTER dataset trimming."""
    actions = np.asarray(actions, dtype=np.int64).ravel()
    total = int(actions.size)
    hold = int(np.count_nonzero(actions == 0))
    play = int(np.count_nonzero(actions != 0))
    from sim.env import ACTIONS, CARD_ACTIONS
    ability = int(np.count_nonzero((actions != 0) & (actions >= CARD_ACTIONS)))
    counts = Counter(int(action) for action in actions if action != 0)
    play_actions = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top10 = play_actions[:10]
    top1_share = float(top10[0][1] / total) if top10 and total else 0.0
    top5_share = float(sum(count for _, count in play_actions[:5]) / total) if total else 0.0
    top10_share = float(sum(count for _, count in play_actions[:10]) / total) if total else 0.0
    return {
        "total": total,
        "hold": hold,
        "play": play,
        "ability": ability,
        "play_actions_distinct": len(counts),
        "top10_actions": [(int(action), int(count)) for action, count in top10],
        "top1_share": top1_share,
        "top5_share": top5_share,
        "top10_share": top10_share,
    }


def action_card_cost(env: ClashEnv, action: int) -> Tuple[Optional[str], int, bool]:
    """Return `(card_name, cost_milli, is_valid_play)` for a play action.

    The card cost is taken from the actual environment data rather than inferred
    from a diff in elixir, which lets the training signal match the simulator's
    real economy.
    """
    if action == 0:
        return None, 0, False
    decoded = ClashEnv.decode(int(action))
    if decoded is None:
        return None, 0, False
    if isinstance(decoded[0], str):
        return None, 0, False
    slot, _, _ = decoded
    hand = env.match.players[1].hand
    if slot >= len(hand):
        return None, 0, False
    card = hand[slot]
    spec = env.match.cards.get(card)
    if spec is None:
        return None, 0, False
    cost = int(getattr(spec, "cost", 0) * 1000)
    return card, cost, True


def cap_action_distribution(actions: list[int], max_per_action: Optional[int]) -> list[int]:
    """Trim only overrepresented actions while preserving rare ones.

    This is intentionally conservative: every rare move remains, and only the
    pencil-thin tail of the most repeated actions is collapsed. It addresses the
    dataset concentration problem without reintroducing the passive V2 hold bias.
    """
    if max_per_action is None or max_per_action <= 0:
        return list(actions)
    counts: Counter[int] = Counter()
    kept: list[int] = []
    for action in actions:
        action_id = int(action)
        if action_id == 0:
            kept.append(action_id)
            continue
        counts[action_id] += 1
        if counts[action_id] <= int(max_per_action):
            kept.append(action_id)
    return kept


def encode_teacher_action(decision, hand, action_mask):
    """Encode one teacher decision and return auditable mapping details."""
    from sim.env import ACTIONS, NUM_SCALARS, ClashEnv

    details = {
        "teacher_card": getattr(decision, "card", None),
        "internal_hand": list(hand),
        "slot": None,
        "encoded_action": 0,
        "mask_valid": False,
        "decoded_action": None,
        "decoded_slot": None,
        "decoded_card": None,
    }
    if decision is None or details["teacher_card"] not in hand:
        return details
    try:
        slot = hand.index(details["teacher_card"])
        action = ClashEnv.encode(slot, int(decision.x), int(decision.y))
    except (AttributeError, TypeError, ValueError):
        return details
    details["slot"] = slot
    details["encoded_action"] = action
    decoded = ClashEnv.decode(action)
    details["decoded_action"] = decoded
    if decoded is None or isinstance(decoded[0], str):
        return details
    decoded_slot, _, _ = decoded
    details["decoded_slot"] = decoded_slot
    if 0 <= action < len(action_mask):
        details["mask_valid"] = bool(action_mask[action])
    if details["mask_valid"] and decoded_slot < len(hand):
        details["decoded_card"] = hand[decoded_slot]
    return details


def summarize_teacher_decision(decision, hand, action_mask):
    """Classify the exact teacher outcome for one decision tick.

    The counters distinguish four kinds of `None`/illegal cases without changing
    the recorded action resolution itself.
    """
    stats = {
        "teacher_none": 0,
        "teacher_decisions": 0,
        "teacher_card_not_in_hand": 0,
        "teacher_invalid_action": 0,
        "teacher_valid_actions": 0,
        "holds_recorded": 0,
        "plays_recorded": 0,
        "decision_kinds": Counter(),
        "decision_cards": Counter(),
    }

    if decision is None:
        stats["teacher_none"] = 1
        return stats

    stats["teacher_decisions"] = 1
    if hasattr(decision, "tag"):
        kind = str(decision.tag)
        stats["decision_kinds"][kind] += 1
    card = getattr(decision, "card", None)
    if card is not None:
        stats["decision_cards"][card] += 1

    if card not in hand:
        stats["teacher_card_not_in_hand"] = 1
        return stats

    mapping = encode_teacher_action(decision, hand, action_mask)
    if mapping["mask_valid"]:
        stats["teacher_valid_actions"] = 1
    else:
        stats["teacher_invalid_action"] = 1
    return stats


def summarize_teacher_stats(stats):
    """Return a compact printable summary for small diagnostic runs."""
    decision_kind_summary = ", ".join(
        f"{kind}:{count}" for kind, count in sorted(stats["decision_kinds"].items())
    ) if stats["decision_kinds"] else "-"
    decision_cards = ", ".join(
        f"{card}:{count}" for card, count in sorted(stats["decision_cards"].items())
    ) if stats["decision_cards"] else "-"
    return (
        f"teacher_none={stats['teacher_none']} "
        f"teacher_decisions={stats['teacher_decisions']} "
        f"teacher_card_not_in_hand={stats['teacher_card_not_in_hand']} "
        f"teacher_invalid_action={stats['teacher_invalid_action']} "
        f"teacher_valid_actions={stats['teacher_valid_actions']} "
        f"raw_teacher_holds={stats['raw_teacher_holds']} "
        f"raw_teacher_plays={stats['raw_teacher_plays']} "
        f"dataset_holds={stats['holds_recorded']} "
        f"dataset_plays={stats['plays_recorded']} "
        f"decision_kinds={decision_kind_summary} "
        f"decision_cards={decision_cards}"
    )


def parse_opponent_mix(spec: str) -> list[tuple[str, float]]:
    """Parse a deterministic weighted opponent mix.

    Format: ``brain:0.5,meta:0.25,mirror:0.25``.
    A single bare name is shorthand for ``name:1.0``.
    The order is preserved and the weights must sum to one.
    """
    allowed = {"brain", "meta", "simple", "mirror"}
    if spec is None:
        spec = "brain:1.0"
    items = []
    for raw in str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            name, weight_text = raw.split(":", 1)
            name = name.strip().lower()
            try:
                weight = float(weight_text)
            except ValueError as exc:
                raise ValueError(f"invalid opponent weight in {raw!r}") from exc
        else:
            name, weight = raw.lower(), 1.0
        if name not in allowed:
            raise ValueError(
                f"unknown opponent {name!r}; choose from "
                f"{', '.join(sorted(allowed))}"
            )
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"opponent weight must be > 0: {raw!r}")
        items.append((name, weight))
    if not items:
        raise ValueError("opponent mix is empty")
    total = sum(weight for _, weight in items)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"opponent mix weights must sum to 1.0, got {total:.6f}"
        )
    return items


def choose_opponent(mix: list[tuple[str, float]], rng: random.Random) -> str:
    """Sample one opponent kind from a fixed weighted mix."""
    pick = rng.random()
    cumulative = 0.0
    for name, weight in mix:
        cumulative += weight
        if pick < cumulative:
            return name
    return mix[-1][0]


def collect(episodes: int, hold_share: Optional[float] = 0.25,
            seed: int = 500_000, diagnostics: bool = False,
            summary_every: Optional[int] = None,
            target_hold_rate: Optional[float] = None,
            max_per_action: Optional[int] = 64,
            elixir_floor: float = 2.0,
            opponent_mix: str = "brain:1.0") -> Tuple[dict, np.ndarray]:
    """Play `episodes` matches with the brain in the agent's seat, recording it.

    A conservative per-action cap is applied after the standard HOLD subsampling.
    This keeps rare legal moves but trims the long tail of the few highly repeated
    actions that otherwise dominate the clone dataset.
    """
    from sim.adapter import build_state
    from sim.env import ACTIONS, NUM_SCALARS, ClashEnv
    from sim.runner import BrainPolicy

    mix = parse_opponent_mix(opponent_mix)
    # `meta` requires the environment to load its deck pool up front. For the
    # default brain-only recipe this remains bit-identical to the known V4
    # collection path.
    env_kind = "meta" if any(name == "meta" for name, _ in mix) else mix[0][0]
    env = ClashEnv(seed=seed, opponent=env_kind)
    teacher = BrainPolicy(env._cards, side=1)

    spill_dir = Path(tempfile.mkdtemp(prefix="hasty_clone_"))
    episode_files = []
    raw_sample_count = 0
    rng = random.Random(0)
    opponent_rng = random.Random(seed ^ 0x5A17BEEF)
    teacher_stats = {
        "teacher_none": 0,
        "teacher_decisions": 0,
        "teacher_card_not_in_hand": 0,
        "teacher_invalid_action": 0,
        "teacher_valid_actions": 0,
        "raw_teacher_holds": 0,
        "raw_teacher_plays": 0,
        "holds_recorded": 0,
        "plays_recorded": 0,
        "decision_kinds": Counter(),
        "decision_cards": Counter(),
        "mapping_samples": 0,
        "mapping_matches": 0,
        "mapping_trace": [],
        "mapping_mismatches": [],
        "opponent_counts": Counter(),
        "hand_rejected_stale": 0,
        "hand_rejected_duplicates": 0,
        "hand_empty_diagnostics": 0,
    }

    for episode in range(episodes):
        raw_samples = []
        opponent_kind = choose_opponent(mix, opponent_rng)
        env.set_opponent_kind(opponent_kind)
        teacher_stats["opponent_counts"][opponent_kind] += 1
        obs, info = env.reset(seed=seed + episode)
        teacher.reset()
        episode_stale = 0
        episode_duplicates = 0
        while True:
            state = build_state(env.match, 1, env._cards)
            now = env.match.elapsed_ms / 1000.0
            decision = teacher.brain.decide(state, now, now)

            action = 0
            if decision is not None:
                hand = env.match.players[1].hand
                mapping = encode_teacher_action(decision, hand, info["action_mask"])
                mapping["state_cards"] = [slot.name for slot in state.cards]
                if mapping["teacher_card"] is not None:
                    teacher_stats["mapping_samples"] += 1
                    mapping_matches = (
                        mapping["mask_valid"]
                        and mapping["decoded_card"] == mapping["teacher_card"]
                    )
                    teacher_stats["mapping_matches"] += int(mapping_matches)
                    if len(teacher_stats["mapping_trace"]) < 20:
                        teacher_stats["mapping_trace"].append(mapping)
                    if not mapping_matches and len(teacher_stats["mapping_mismatches"]) < 20:
                        teacher_stats["mapping_mismatches"].append(mapping)
                step_stats = summarize_teacher_decision(decision, hand, info["action_mask"])
                for key in ("teacher_none", "teacher_decisions", "teacher_card_not_in_hand",
                            "teacher_invalid_action", "teacher_valid_actions"):
                    teacher_stats[key] += step_stats[key]
                teacher_stats["decision_kinds"].update(step_stats["decision_kinds"])
                teacher_stats["decision_cards"].update(step_stats["decision_cards"])
                if step_stats["teacher_valid_actions"]:
                    candidate = mapping["encoded_action"]
                    if 0 <= candidate < len(info["action_mask"]) \
                            and info["action_mask"][candidate]:
                        action = candidate
                    else:
                        # The brain sometimes wants a tile the engine will not
                        # accept. Recording it would teach an illegal habit.
                        rejected = 0

            card = None
            cost = 0
            elixir_before = int(env.match.players[1].elixir)
            sustainable = False
            if action != 0:
                card, cost, valid = action_card_cost(env, action)
                if valid:
                    sustainable = (elixir_before - cost) >= int(elixir_floor * 1000)
            raw_samples.append({
                "planes": obs["planes"],
                "scalars": obs["scalars"],
                "mask": info["action_mask"],
                "action": action,
                "episode_id": episode,
                "sample_index": raw_sample_count,
                "card": card,
                "cost": cost,
                "elixir_before": elixir_before,
                "sustainable": sustainable,
            })
            teacher_stats["raw_teacher_holds"] += action == 0
            teacher_stats["raw_teacher_plays"] += action != 0
            raw_sample_count += 1

            obs, _, terminated, truncated, info = env.step(action)
            if action != 0 and decision is not None:
                teacher.brain.confirm(decision, now)
            if terminated or truncated:
                break
        if diagnostics and (summary_every is None or (episode + 1) % summary_every == 0):
            rss = process_rss_mb()
            rss_text = f" rss_mb={rss:.1f}" if rss is not None else ""
            log(f"teacher diag: {summarize_teacher_stats(teacher_stats)}{rss_text} "
                f"raw_samples_retained=0 episodes={episode + 1}")

        episode_stale = teacher.brain.hand_rejected_stale
        episode_duplicates = teacher.brain.hand_rejected_duplicates
        teacher_stats["hand_rejected_stale"] += episode_stale
        teacher_stats["hand_rejected_duplicates"] += episode_duplicates
        teacher_stats["hand_empty_diagnostics"] += int(
            teacher.brain.hand_empty_diagnostic_emitted)

        episode_path = spill_dir / f"episode_{episode:05d}.npz"
        np.savez(
            episode_path,
            planes=np.stack([entry["planes"] for entry in raw_samples]),
            scalars=np.stack([entry["scalars"] for entry in raw_samples]),
            masks=np.stack([entry["mask"] for entry in raw_samples]),
            actions=np.asarray([entry["action"] for entry in raw_samples], dtype=np.int64),
            sustainable=np.asarray([entry["sustainable"] for entry in raw_samples], dtype=bool),
            episode_id=np.asarray([entry["episode_id"] for entry in raw_samples], dtype=np.int64),
            sample_index=np.asarray([entry["sample_index"] for entry in raw_samples], dtype=np.int64),
        )
        episode_files.append(episode_path)
        del raw_samples

    raw_teacher_total = teacher_stats["raw_teacher_holds"] + teacher_stats["raw_teacher_plays"]
    if raw_teacher_total == 0:
        effective_hold_share = 0.0 if hold_share is None else float(hold_share)
    elif hold_share is None:
        raw_hold_rate = teacher_stats["raw_teacher_holds"] / raw_teacher_total
        target_hold_rate = 0.30 if target_hold_rate is None else target_hold_rate
        effective_hold_share = estimate_hold_keep_rate(raw_hold_rate, target_hold_rate)
    else:
        effective_hold_share = float(hold_share)

    filtered_actions = []
    kept_by_action = Counter()
    selected_flags = []
    for episode_path in episode_files:
        with np.load(episode_path) as episode_data:
            flags = np.zeros(len(episode_data["actions"]), dtype=bool)
            for index, action_value in enumerate(episode_data["actions"]):
                action = int(action_value)
                if action == 0 and rng.random() >= effective_hold_share:
                    continue
                filtered_actions.append(action)
                if action != 0:
                    kept_by_action[action] += 1
                    if max_per_action is not None and max_per_action > 0 \
                            and kept_by_action[action] > int(max_per_action):
                        continue
                flags[index] = True
            selected_flags.append(flags)

    selected_count = int(sum(int(flags.sum()) for flags in selected_flags))
    with np.load(episode_files[0]) as first_episode:
        plane_shape = first_episode["planes"].shape[1:]
    planes = np.empty((selected_count,) + tuple(plane_shape), dtype=np.float32)
    scalars = np.empty((selected_count, NUM_SCALARS), dtype=np.float32)
    masks = np.empty((selected_count, ACTIONS), dtype=bool)
    actions = np.empty(selected_count, dtype=np.int64)
    sustainable = np.empty(selected_count, dtype=bool)
    episode_ids = np.empty(selected_count, dtype=np.int64)
    sample_indices = np.empty(selected_count, dtype=np.int64)
    output_index = 0
    for episode_path, flags in zip(episode_files, selected_flags):
        with np.load(episode_path) as episode_data:
            selected = np.flatnonzero(flags)
            end = output_index + len(selected)
            planes[output_index:end] = episode_data["planes"][selected]
            scalars[output_index:end] = episode_data["scalars"][selected]
            masks[output_index:end] = episode_data["masks"][selected]
            actions[output_index:end] = episode_data["actions"][selected]
            sustainable[output_index:end] = episode_data["sustainable"][selected]
            episode_ids[output_index:end] = episode_data["episode_id"][selected]
            sample_indices[output_index:end] = episode_data["sample_index"][selected]
            output_index = end
    teacher_stats["holds_recorded"] = int(np.count_nonzero(actions == 0))
    teacher_stats["plays_recorded"] = int(np.count_nonzero(actions != 0))

    before_stats = summarize_action_distribution(filtered_actions)
    after_stats = summarize_action_distribution(actions)

    env.close()
    for episode_path in episode_files:
        episode_path.unlink(missing_ok=True)
    spill_dir.rmdir()
    teacher_stats["hold_share_used"] = effective_hold_share
    teacher_stats["max_per_action"] = int(max_per_action) if max_per_action is not None else None
    teacher_stats["dataset_before"] = before_stats
    teacher_stats["dataset_after"] = after_stats
    teacher_stats["raw_teacher_hold_rate"] = (
        teacher_stats["raw_teacher_holds"] / max(raw_teacher_total, 1)
    )
    teacher_stats["dataset_hold_rate"] = (
        teacher_stats["holds_recorded"] / max(len(actions), 1)
    )
    teacher_stats["dataset_play_rate"] = (
        teacher_stats["plays_recorded"] / max(len(actions), 1)
    )
    teacher_stats["target_hold_rate"] = target_hold_rate
    teacher_stats["teacher_action_mapping_accuracy"] = (
        teacher_stats["mapping_matches"] / max(teacher_stats["mapping_samples"], 1)
    )
    data = {
        "planes": planes,
        "scalars": scalars,
        "masks": masks,
        "sustainable": sustainable,
        "episode_id": episode_ids,
        "sample_index": sample_indices,
        "teacher_stats": teacher_stats,
    }
    return data, actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Behaviour-clone the brain")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hold-share", type=float, default=None,
                        help="HOLD retention rate. If omitted, compute it from the raw teacher rate and --target-hold-rate.")
    parser.add_argument("--target-hold-rate", type=float, default=0.30,
                        help="Target final HOLD rate for the dataset, used with auto hold-share estimation.")
    parser.add_argument("--name", default="clone")
    parser.add_argument("--max-per-action", type=int, default=64,
                        help="conservative cap on repeated PLAY actions; rare actions are preserved")
    parser.add_argument("--play-entropy-coef", type=float, default=0.005,
                        help="small entropy regularizer applied only on legal PLAY logits")
    parser.add_argument("--elixir-floor", type=float, default=2.0,
                        help="minimum elixir left after a legal PLAY for the action to count as sustainable")
    parser.add_argument("--opponent-mix", default="brain:1.0",
                        help="weighted teacher-collection opponent mix, e.g. "
                             "brain:0.5,meta:0.25,mirror:0.25. Default preserves "
                             "the V4 brain-only recipe.")
    parser.add_argument("--sustainability-weight", type=float, default=0.0,
                        help="0.0 (default) = sustainability OFF: trains on every teacher PLAY, "
                             "matching the pre-sustainability recipe. >0.0 = ON: reclassifies "
                             "non-sustainable teacher PLAYs as HOLD in the loss (this is the "
                             "experimental behaviour that produced clone_v4_test2b.pt and scored "
                             "worse than V4). Kept OFF by default so a plain `python -m sim.clone` "
                             "run reproduces the known-good recipe instead of the test2b one.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="print compact teacher decision counters")
    parser.add_argument("--summary-every", type=int, default=1,
                        help="emit the compact teacher summary every N episodes")
    parser.add_argument("--init", type=Path,
                        help="start from a checkpoint state_dict; weights only, no optimiser state")
    parser.add_argument("--save-diagnostic-batch", type=Path,
                        help="persist the collected batch as a temporary .npz for later diagnostics")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="save the diagnostic batch and exit before training/evaluation")
    parser.add_argument("--load-diagnostic-batch", type=Path,
                        help="load a persisted diagnostic batch instead of collecting episodes")
    args = parser.parse_args()

    if args.diagnostic_only and args.save_diagnostic_batch is None:
        parser.error("--diagnostic-only requires --save-diagnostic-batch")
    if args.load_diagnostic_batch is not None and (args.diagnostic_only or args.save_diagnostic_batch is not None):
        parser.error("--load-diagnostic-batch cannot be combined with diagnostic capture flags")

    import torch
    import torch.nn as nn
    from sim.env import ACTIONS, ABILITY_SLOTS, CARD_ACTIONS, NUM_PLANES, NUM_SCALARS
    from sim.train_ppo import build_network, hierarchical_action_from_logits, masked_distribution

    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.sustainability_weight > 0.0:
        log(f"SUSTAINABILITY: ON (elixir_floor={args.elixir_floor}) -- experimental recipe, "
            f"matches what produced clone_v4_test2b.pt")
    else:
        log("SUSTAINABILITY: OFF -- default recipe, trains on every teacher PLAY")
    loaded_metadata = None
    if args.load_diagnostic_batch is not None:
        data, actions, loaded_metadata = load_or_collect_batch(args.load_diagnostic_batch)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("loaded diagnostic batch must contain non-empty train_idx and val_idx")
        if np.any(train_idx < 0) or np.any(val_idx < 0) \
                or np.any(train_idx >= len(actions)) or np.any(val_idx >= len(actions)):
            raise ValueError("loaded diagnostic batch contains out-of-range split indices")
        log(f"loaded diagnostic batch {args.load_diagnostic_batch} "
            f"({len(actions)} samples, train={len(train_idx)}, val={len(val_idx)})")
    else:
        data, actions = load_or_collect_batch(
            diagnostics=args.diagnostics,
            episodes=args.episodes,
            hold_share=args.hold_share,
            summary_every=args.summary_every,
            target_hold_rate=args.target_hold_rate,
            max_per_action=args.max_per_action,
            elixir_floor=args.elixir_floor,
            opponent_mix=args.opponent_mix)
    sustainability = torch.from_numpy(data["sustainable"]).to(device)
    played = int((actions != 0).sum())
    teacher_stats = data.get("teacher_stats")
    if teacher_stats is not None:
        mix_summary = ", ".join(
            f"{name}:{count}" for name, count in sorted(teacher_stats["opponent_counts"].items())
        ) or "-"
        log(f"collection opponents: {mix_summary} (target mix={args.opponent_mix})")
        log(f"teacher raw HOLD {teacher_stats['raw_teacher_holds']}/{teacher_stats['raw_teacher_holds'] + teacher_stats['raw_teacher_plays']} "
            f"({teacher_stats['raw_teacher_hold_rate']:.2%})  dataset HOLD {teacher_stats['holds_recorded']}/{len(actions)} "
            f"({teacher_stats['dataset_hold_rate']:.2%})  dataset PLAY {teacher_stats['plays_recorded']}/{len(actions)} "
            f"({teacher_stats['dataset_play_rate']:.2%})  hold_share={teacher_stats['hold_share_used']:.3f}")
        log(f"teacher_action_mapping_accuracy={teacher_stats['teacher_action_mapping_accuracy']:.2%} "
            f"({teacher_stats['mapping_matches']}/{teacher_stats['mapping_samples']})")
    log(f"dataset {len(actions)} samples ({played} plays, "
        f"{len(actions) - played} holds) in {time.time() - started:.0f}s")

    network = build_network(NUM_PLANES, NUM_SCALARS, ACTIONS).to(device)
    if args.init is not None:
        if not args.init.exists():
            raise SystemExit(f"--init {args.init} does not exist")
        blob = torch.load(args.init, map_location=device, weights_only=False)
        if "state_dict" not in blob:
            raise SystemExit(f"--init {args.init} does not contain a state_dict")
        network.load_state_dict(blob["state_dict"])
        log(f"initialised from {args.init} (weights only)")
    optimiser = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4)

    count = len(actions)
    if args.load_diagnostic_batch is not None:
        split_metadata = {
            "split_mode": loaded_metadata.get("split_mode", "persisted")
            if loaded_metadata is not None else "persisted",
        }
    elif args.diagnostic_only:
        episode_ids = data["episode_id"]
        episodes_in_order = np.unique(episode_ids)
        if len(episodes_in_order) < 2:
            raise ValueError("diagnostic-only episode split requires at least two episodes")
        train_episode_count = max(1, int(np.floor(len(episodes_in_order) * 0.8)))
        train_episodes = episodes_in_order[:train_episode_count]
        val_episodes = episodes_in_order[train_episode_count:]
        train_idx = np.flatnonzero(np.isin(episode_ids, train_episodes))
        val_idx = np.flatnonzero(np.isin(episode_ids, val_episodes))
        split_metadata = {
            "split_mode": "episode_order",
            "train_episodes": train_episodes.tolist(),
            "val_episodes": val_episodes.tolist(),
        }
    else:
        split = int(count * 0.9)
        order = np.random.default_rng(0).permutation(count)
        train_idx, val_idx = order[:split], order[split:]
        split_metadata = {
            "split_mode": "sample_random",
            "split_seed": 0,
        }

    if args.save_diagnostic_batch is not None:
        save_diagnostic_batch(
            args.save_diagnostic_batch,
            data,
            actions,
            train_idx=train_idx,
            val_idx=val_idx,
            metadata={
                "source": "sim.clone.collect",
                "seed": 500_000,
                "episodes": args.episodes,
                "hold_share": args.hold_share,
                "target_hold_rate": args.target_hold_rate,
                "max_per_action": args.max_per_action,
                "elixir_floor": args.elixir_floor,
                "opponent_mix": args.opponent_mix,
                "split_fraction": 0.8 if args.diagnostic_only else 0.9,
                **split_metadata,
                "play_entropy_coef": args.play_entropy_coef,
                "sustainability_weight": args.sustainability_weight,
                "action_count": ACTIONS,
                "card_actions": CARD_ACTIONS,
                "ability_slots": ABILITY_SLOTS,
                "teacher_stats": {
                    key: teacher_stats[key]
                    for key in ("raw_teacher_hold_rate", "dataset_hold_rate",
                                "dataset_play_rate", "hold_share_used",
                                "teacher_action_mapping_accuracy")
                },
                "episode_distribution": [
                    {
                        "episode_id": int(episode_id),
                        "samples": int(np.count_nonzero(data["episode_id"] == episode_id)),
                        "hold": int(np.count_nonzero(
                            actions[data["episode_id"] == episode_id] == 0)),
                        "play": int(np.count_nonzero(
                            actions[data["episode_id"] == episode_id] != 0)),
                    }
                    for episode_id in np.unique(data["episode_id"])
                ],
            },
        )
        log(f"saved diagnostic batch {args.save_diagnostic_batch}")
        if args.diagnostic_only:
            return 0

    def batch_of(index: np.ndarray):
        return (torch.from_numpy(data["planes"][index]).to(device),
                torch.from_numpy(data["scalars"][index]).to(device),
                torch.from_numpy(data["masks"][index]).to(device),
                torch.from_numpy(actions[index]).to(device))

    criterion = nn.CrossEntropyLoss()
    best = (-float("inf"), -float("inf"), -float("inf"))
    best_epoch = None
    best_metrics = None
    for epoch in range(args.epochs):
        network.train()
        np.random.shuffle(train_idx)
        total = 0.0
        for start in range(0, len(train_idx), args.batch):
            batch_idx = train_idx[start:start + args.batch]
            planes, scalars, masks, target = batch_of(batch_idx)
            batch_sustainability = sustainability[torch.as_tensor(batch_idx, device=device, dtype=torch.long)]
            logits, value = network(planes, scalars)
            sustainability_on = args.sustainability_weight > 0.0
            if sustainability_on:
                # Experimental mode: a teacher PLAY that would leave less than
                # --elixir-floor elixir is relabelled as HOLD in the loss. This
                # is the exact recipe that produced clone_v4_test2b.pt, which
                # scored worse than V4 (reward -34.17 vs -28.85, own_damage
                # 99.66% vs 82.54%). Opt-in only: pass --sustainability-weight
                # > 0 to use it, and treat the result as a new named experiment,
                # not as the default recipe.
                train_target = torch.where(
                    (target != 0) & (~batch_sustainability),
                    torch.zeros_like(target),
                    target,
                )
            else:
                # Default: plain HOLD-vs-PLAY classification against every
                # teacher decision, with no elixir-based relabelling. This is
                # the recipe believed to have produced clone_v4_probe_init.pt.
                train_target = target
            loss_parts = hierarchical_clone_loss(logits, masks, train_target)
            loss = loss_parts["loss"]
            if args.play_entropy_coef > 0.0:
                play_mask = masks[:, 1:CARD_ACTIONS]
                legal_rows = (train_target != 0) & play_mask.any(dim=1)
                if legal_rows.any():
                    play_logits = logits[:, 1:CARD_ACTIONS][legal_rows].masked_fill(
                        ~play_mask[legal_rows], float("-inf"))
                    play_dist = torch.softmax(play_logits, dim=-1)
                    play_entropy = -(play_dist * torch.log(play_dist.clamp_min(1e-8))).sum(dim=-1).mean()
                    loss = loss - args.play_entropy_coef * play_entropy
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimiser.step()
            total += loss.detach().item() * len(target)

        network.eval()
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for start in range(0, len(val_idx), 1024):
                planes, scalars, masks, target = batch_of(val_idx[start:start + 1024])
                logits, _ = network(planes, scalars)
                predicted = hierarchical_action_from_logits(logits, masks)
                val_targets.append(target.cpu().numpy())
                val_predictions.append(predicted.cpu().numpy())
        val_targets = np.concatenate(val_targets) if val_targets else np.array([], dtype=np.int64)
        val_predictions = np.concatenate(val_predictions) if val_predictions else np.array([], dtype=np.int64)
        metrics = compute_validation_metrics(val_targets, val_predictions)
        accuracy = metrics["clone_accuracy"]
        selection_key = checkpoint_selection_key(metrics)
        log(f"epoch {epoch + 1}/{args.epochs}  loss {total / len(train_idx):.4f}  "
            f"val {accuracy:.3f}  always_hold {metrics['always_hold_accuracy']:.3f}  "
            f"hold {metrics['hold_accuracy']:.3f}  nonhold {metrics['nonhold_accuracy']:.3f}  "
            f"play {metrics['play_accuracy']:.3f}  ability {metrics['ability_accuracy']:.3f}  "
            f"macro {metrics['macro_accuracy']:.3f}  wait {metrics['validation_wait_rate']:.3f}  "
            f"nonwait {metrics['validation_nonwait_rate']:.3f}  gate {metrics['gate_accuracy']:.3f}  "
            f"nonhold_pred {metrics['nonhold_classes_predicted']}")
        if selection_key > best:
            best = selection_key
            best_epoch = epoch + 1
            best_metrics = dict(metrics)
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": network.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "step": 0,
                        **checkpoint_metadata(epoch + 1, metrics, selection_key),
                        "val_accuracy": accuracy,
                        "val_macro_accuracy": metrics["macro_accuracy"],
                        "val_gate_accuracy": metrics["gate_accuracy"],
                        "val_nonhold_accuracy": metrics["nonhold_accuracy"],
                        "val_play_accuracy": metrics["play_accuracy"],
                        "val_ability_accuracy": metrics["ability_accuracy"]},
                       OUT / f"{args.name}.pt")

    log(f"best validation selection best_epoch={best_epoch} selection_key={best} "
        f"gate_gain={best_metrics['gate_accuracy'] - best_metrics['always_hold_accuracy']:.3f} "
        f"nonhold={best_metrics['nonhold_accuracy']:.3f} play={best_metrics['play_accuracy']:.3f} -> "
        f"{OUT / (args.name + '.pt')}")

    from sim.train_ppo import evaluate
    blob = torch.load(OUT / f"{args.name}.pt", map_location=device, weights_only=False)
    network.load_state_dict(blob["state_dict"])
    network.eval()
    result = evaluate(network, device, episodes=16,
                      action_selector=hierarchical_action_from_logits)
    log(f"EVAL(clone)  W{result['wins']} L{result['losses']} D{result['draws']}  "
        f"crowns {result['crowns_for']}-{result['crowns_against']}  "
        f"hog {result['hog_share']:.0%}  plays/match {result['plays_per_match']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
