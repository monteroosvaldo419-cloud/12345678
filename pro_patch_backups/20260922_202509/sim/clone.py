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
import random
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


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} clone: {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


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
    """Metrics for HOLD-vs-PLAY collapse detection on a real validation set."""
    target = np.asarray(target, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if target.size == 0:
        return {
            "always_hold_accuracy": 0.0,
            "clone_accuracy": 0.0,
            "hold_accuracy": 0.0,
            "play_accuracy": 0.0,
            "macro_accuracy": 0.0,
            "validation_wait_rate": 0.0,
            "validation_nonwait_rate": 0.0,
            "nonhold_classes_predicted": 0,
        }

    hold_mask = target == 0
    play_mask = target != 0
    always_hold = np.zeros_like(target)
    always_hold_accuracy = float((always_hold == target).mean())
    clone_accuracy = float((predicted == target).mean())
    hold_accuracy = float((predicted[hold_mask] == target[hold_mask]).mean()) if hold_mask.any() else 0.0
    play_accuracy = float((predicted[play_mask] == target[play_mask]).mean()) if play_mask.any() else 0.0
    macro_accuracy = float((hold_accuracy + play_accuracy) / 2.0) if hold_mask.any() and play_mask.any() else clone_accuracy
    validation_wait_rate = float((predicted == 0).mean())
    validation_nonwait_rate = float((predicted != 0).mean())
    nonhold_classes_predicted = int(np.unique(predicted[predicted != 0]).size) if np.any(predicted != 0) else 0
    return {
        "always_hold_accuracy": always_hold_accuracy,
        "clone_accuracy": clone_accuracy,
        "hold_accuracy": hold_accuracy,
        "play_accuracy": play_accuracy,
        "macro_accuracy": macro_accuracy,
        "validation_wait_rate": validation_wait_rate,
        "validation_nonwait_rate": validation_nonwait_rate,
        "nonhold_classes_predicted": nonhold_classes_predicted,
    }


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

    try:
        slot = hand.index(card)
        candidate = int(decision.x)
        target_x = int(decision.y)
    except (TypeError, ValueError, AttributeError):
        stats["teacher_invalid_action"] = 1
        return stats

    try:
        from sim.env import ClashEnv
        action_index = ClashEnv.encode(slot, candidate, target_x)
    except Exception:
        stats["teacher_invalid_action"] = 1
        return stats

    if 0 <= action_index < len(action_mask) and bool(action_mask[action_index]):
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
    from sim.env import ClashEnv
    from sim.runner import BrainPolicy

    mix = parse_opponent_mix(opponent_mix)
    # `meta` requires the environment to load its deck pool up front. For the
    # default brain-only recipe this remains bit-identical to the known V4
    # collection path.
    env_kind = "meta" if any(name == "meta" for name, _ in mix) else mix[0][0]
    env = ClashEnv(seed=seed, opponent=env_kind)
    teacher = BrainPolicy(env._cards, side=1)

    raw_samples = []
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
        "opponent_counts": Counter(),
    }

    for episode in range(episodes):
        opponent_kind = choose_opponent(mix, opponent_rng)
        env.set_opponent_kind(opponent_kind)
        teacher_stats["opponent_counts"][opponent_kind] += 1
        obs, info = env.reset(seed=seed + episode)
        teacher.reset()
        while True:
            state = build_state(env.match, 1, env._cards)
            now = env.match.elapsed_ms / 1000.0
            decision = teacher.brain.decide(state, now, now)

            action = 0
            if decision is not None:
                hand = env.match.players[1].hand
                step_stats = summarize_teacher_decision(decision, hand, info["action_mask"])
                for key in ("teacher_none", "teacher_decisions", "teacher_card_not_in_hand",
                            "teacher_invalid_action", "teacher_valid_actions"):
                    teacher_stats[key] += step_stats[key]
                teacher_stats["decision_kinds"].update(step_stats["decision_kinds"])
                teacher_stats["decision_cards"].update(step_stats["decision_cards"])
                if step_stats["teacher_valid_actions"]:
                    slot = hand.index(decision.card)
                    candidate = ClashEnv.encode(slot, int(decision.x), int(decision.y))
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
                "card": card,
                "cost": cost,
                "elixir_before": elixir_before,
                "sustainable": sustainable,
            })
            teacher_stats["raw_teacher_holds"] += action == 0
            teacher_stats["raw_teacher_plays"] += action != 0

            obs, _, terminated, truncated, info = env.step(action)
            if action != 0 and decision is not None:
                teacher.brain.confirm(decision, now)
            if terminated or truncated:
                break
        if diagnostics and (summary_every is None or (episode + 1) % summary_every == 0):
            log(f"teacher diag: {summarize_teacher_stats(teacher_stats)}")

    raw_teacher_total = teacher_stats["raw_teacher_holds"] + teacher_stats["raw_teacher_plays"]
    if raw_teacher_total == 0:
        effective_hold_share = 0.0 if hold_share is None else float(hold_share)
    elif hold_share is None:
        raw_hold_rate = teacher_stats["raw_teacher_holds"] / raw_teacher_total
        target_hold_rate = 0.30 if target_hold_rate is None else target_hold_rate
        effective_hold_share = estimate_hold_keep_rate(raw_hold_rate, target_hold_rate)
    else:
        effective_hold_share = float(hold_share)

    filtered_entries = []
    for entry in raw_samples:
        keep = entry["action"] != 0 or rng.random() < effective_hold_share
        if keep:
            filtered_entries.append(entry)

    before_stats = summarize_action_distribution([entry["action"] for entry in filtered_entries])
    kept_by_action = Counter()
    selected_entries = []
    for entry in filtered_entries:
        action = int(entry["action"])
        if action == 0:
            selected_entries.append(entry)
            continue
        kept_by_action[action] += 1
        if kept_by_action[action] <= int(max_per_action or 0):
            selected_entries.append(entry)

    after_stats = summarize_action_distribution([entry["action"] for entry in selected_entries])

    planes: List[np.ndarray] = []
    scalars: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    actions: List[int] = []
    sustainable: List[bool] = []
    for entry in selected_entries:
        planes.append(entry["planes"])
        scalars.append(entry["scalars"])
        masks.append(entry["mask"])
        actions.append(entry["action"])
        sustainable.append(bool(entry.get("sustainable", False)))
        teacher_stats["holds_recorded"] += entry["action"] == 0
        teacher_stats["plays_recorded"] += entry["action"] != 0

    env.close()
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
    data = {
        "planes": np.stack(planes),
        "scalars": np.stack(scalars),
        "masks": np.stack(masks),
        "sustainable": np.asarray(sustainable, dtype=bool),
        "teacher_stats": teacher_stats,
    }
    return data, np.array(actions, dtype=np.int64)


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
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from sim.env import ACTIONS, CARD_ACTIONS, NUM_PLANES, NUM_SCALARS
    from sim.train_ppo import build_network, hierarchical_action_from_logits, masked_distribution

    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.sustainability_weight > 0.0:
        log(f"SUSTAINABILITY: ON (elixir_floor={args.elixir_floor}) -- experimental recipe, "
            f"matches what produced clone_v4_test2b.pt")
    else:
        log("SUSTAINABILITY: OFF -- default recipe, trains on every teacher PLAY")
    data, actions = collect(args.episodes, args.hold_share,
                            diagnostics=args.diagnostics,
                            summary_every=args.summary_every,
                            target_hold_rate=args.target_hold_rate,
                            max_per_action=args.max_per_action,
                            elixir_floor=args.elixir_floor,
                            opponent_mix=args.opponent_mix)
    sustainability = torch.from_numpy(data["sustainable"]).to(device)
    played = int((actions != 0).sum())
    teacher_stats = data["teacher_stats"]
    mix_summary = ", ".join(
        f"{name}:{count}" for name, count in sorted(teacher_stats["opponent_counts"].items())
    ) or "-"
    log(f"collection opponents: {mix_summary} (target mix={args.opponent_mix})")
    log(f"teacher raw HOLD {teacher_stats['raw_teacher_holds']}/{teacher_stats['raw_teacher_holds'] + teacher_stats['raw_teacher_plays']} "
        f"({teacher_stats['raw_teacher_hold_rate']:.2%})  dataset HOLD {teacher_stats['holds_recorded']}/{len(actions)} "
        f"({teacher_stats['dataset_hold_rate']:.2%})  dataset PLAY {teacher_stats['plays_recorded']}/{len(actions)} "
        f"({teacher_stats['dataset_play_rate']:.2%})  hold_share={teacher_stats['hold_share_used']:.3f}")
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
    split = int(count * 0.9)
    order = np.random.default_rng(0).permutation(count)
    train_idx, val_idx = order[:split], order[split:]

    def batch_of(index: np.ndarray):
        return (torch.from_numpy(data["planes"][index]).to(device),
                torch.from_numpy(data["scalars"][index]).to(device),
                torch.from_numpy(data["masks"][index]).to(device),
                torch.from_numpy(actions[index]).to(device))

    criterion = nn.CrossEntropyLoss()
    best = (-float("inf"), -float("inf"))
    for epoch in range(args.epochs):
        network.train()
        np.random.shuffle(train_idx)
        total = 0.0
        for start in range(0, len(train_idx), args.batch):
            batch_idx = train_idx[start:start + args.batch]
            planes, scalars, masks, target = batch_of(batch_idx)
            batch_sustainability = sustainability[torch.as_tensor(batch_idx, device=device, dtype=torch.long)]
            logits, value = network(planes, scalars)
            hold_logits = logits[:, 0]
            sustainability_on = args.sustainability_weight > 0.0
            if sustainability_on:
                # Experimental mode: a teacher PLAY that would leave less than
                # --elixir-floor elixir is relabelled as HOLD in the loss. This
                # is the exact recipe that produced clone_v4_test2b.pt, which
                # scored worse than V4 (reward -34.17 vs -28.85, own_damage
                # 99.66% vs 82.54%). Opt-in only: pass --sustainability-weight
                # > 0 to use it, and treat the result as a new named experiment,
                # not as the default recipe.
                sustain_target = ((target == 0) | ((target != 0) & (~batch_sustainability))).float()
            else:
                # Default: plain HOLD-vs-PLAY classification against every
                # teacher decision, with no elixir-based relabelling. This is
                # the recipe believed to have produced clone_v4_probe_init.pt.
                sustain_target = (target == 0).float()
            hold_loss = F.binary_cross_entropy_with_logits(hold_logits, sustain_target)

            play_logits = logits[:, 1:CARD_ACTIONS]
            play_mask = masks[:, 1:CARD_ACTIONS]
            play_logits = play_logits.masked_fill(~play_mask, float("-inf"))
            if sustainability_on:
                play_keep = (target > 0) & (target < CARD_ACTIONS) & batch_sustainability
            else:
                play_keep = (target > 0) & (target < CARD_ACTIONS)
            if play_keep.any():
                play_target = target[play_keep] - 1
                play_loss = F.cross_entropy(play_logits[play_keep], play_target)
                if args.play_entropy_coef > 0.0:
                    legal_rows = play_mask.any(dim=1)
                    if legal_rows.any():
                        legal_logits = play_logits[legal_rows].masked_fill(~play_mask[legal_rows], float("-inf"))
                        legal_dist = torch.softmax(legal_logits, dim=-1)
                        play_entropy = -(legal_dist * torch.log(legal_dist.clamp_min(1e-8))).sum(dim=-1).mean()
                        play_loss = play_loss - args.play_entropy_coef * play_entropy
            else:
                play_loss = torch.zeros((), device=planes.device, dtype=logits.dtype)
            loss = hold_loss + play_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimiser.step()
            total += float(loss) * len(target)

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
        selection_key = (metrics["macro_accuracy"], metrics["play_accuracy"])
        log(f"epoch {epoch + 1}/{args.epochs}  loss {total / len(train_idx):.4f}  "
            f"val {accuracy:.3f}  always_hold {metrics['always_hold_accuracy']:.3f}  "
            f"hold {metrics['hold_accuracy']:.3f}  play {metrics['play_accuracy']:.3f}  "
            f"macro {metrics['macro_accuracy']:.3f}  wait {metrics['validation_wait_rate']:.3f}  "
            f"nonwait {metrics['validation_nonwait_rate']:.3f}  nonhold_pred {metrics['nonhold_classes_predicted']}")
        if selection_key > best:
            best = selection_key
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": network.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "step": 0,
                        "val_accuracy": accuracy,
                        "val_macro_accuracy": metrics["macro_accuracy"],
                        "val_play_accuracy": metrics["play_accuracy"],
                        "selection_key": selection_key},
                       OUT / f"{args.name}.pt")

    log(f"best validation selection macro={best[0]:.3f} play={best[1]:.3f} -> "
        f"{OUT / (args.name + '.pt')}")

    from sim.train_ppo import evaluate
    blob = torch.load(OUT / f"{args.name}.pt", map_location=device, weights_only=False)
    network.load_state_dict(blob["state_dict"])
    network.eval()
    result = evaluate(network, device, episodes=16)
    log(f"EVAL(clone)  W{result['wins']} L{result['losses']} D{result['draws']}  "
        f"crowns {result['crowns_for']}-{result['crowns_against']}  "
        f"hog {result['hog_share']:.0%}  plays/match {result['plays_per_match']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



