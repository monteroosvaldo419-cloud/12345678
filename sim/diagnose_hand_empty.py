"""Solo-lectura: mide cuanto se corrompe el dataset de clonado por el bug de
'mano percibida vacia'. No entrena nada, no guarda ningun checkpoint.

Uso:
    python -m sim.diagnose_hand_empty --episodes 100

(Copiar este archivo a la carpeta sim/ antes de correrlo, ver instrucciones
en la respuesta del chat.)
"""
from __future__ import annotations

import argparse

from sim.adapter import build_state
from sim.env import ClashEnv
from sim.runner import BrainPolicy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=500_000)
    args = parser.parse_args()

    env = ClashEnv(seed=args.seed)
    teacher = BrainPolicy(env._cards, side=1)

    total_ticks = 0
    empty_perceived_hand = 0
    empty_but_real_hand_nonempty = 0
    real_hand_playable_ignored = 0
    decisions_none = 0

    for episode in range(args.episodes):
        env.reset(seed=args.seed + episode)
        teacher.reset()
        while True:
            state = build_state(env.match, 1, env._cards)
            now = env.match.elapsed_ms / 1000.0
            decision = teacher.brain.decide(state, now, now)
            total_ticks += 1

            perceived_hand = getattr(teacher.brain.last_obs, "hand", {}) or {}
            real_hand = env.match.players[1].hand
            real_elixir = env.match.players[1].elixir

            if not perceived_hand:
                empty_perceived_hand += 1
                if real_hand:
                    empty_but_real_hand_nonempty += 1
                    # De esas, cuantas tenian ademas elixir de sobra para
                    # jugar al menos la carta mas barata del mazo (2000 = 2.0)
                    if real_elixir >= 2000:
                        real_hand_playable_ignored += 1

            if decision is None:
                decisions_none += 1

            action = 0
            if decision is not None:
                hand = env.match.players[1].hand
                if decision.card in hand:
                    slot = hand.index(decision.card)
                    candidate = ClashEnv.encode(slot, int(decision.x), int(decision.y))
                    action = candidate

            obs2, _, terminated, truncated, info = env.step(action)
            if action != 0 and decision is not None:
                teacher.brain.confirm(decision, now)
            if terminated or truncated:
                break

    env.close()

    print(f"episodios={args.episodes}")
    print(f"ticks_totales={total_ticks}")
    print(f"mano_percibida_vacia={empty_perceived_hand} "
          f"({100.0 * empty_perceived_hand / max(1, total_ticks):.2f}% de los ticks)")
    print(f"  de esas, mano_real_NO_vacia={empty_but_real_hand_nonempty} "
          f"({100.0 * empty_but_real_hand_nonempty / max(1, empty_perceived_hand):.2f}% de las manos-vacias-percibidas)")
    print(f"  de esas, con elixir de sobra para jugar algo="
          f"{real_hand_playable_ignored}")
    print(f"decisiones_None_del_maestro={decisions_none} "
          f"({100.0 * decisions_none / max(1, total_ticks):.2f}% de los ticks)")
    if decisions_none:
        print(f"  de las decisiones_None, causadas por mano-vacia-erronea="
              f"{empty_but_real_hand_nonempty} "
              f"({100.0 * empty_but_real_hand_nonempty / decisions_none:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
