# Simulator Architecture

## Scope

The repository already contains the causal simulator under `sim/`. It is
headless Python state, independent from ADB, PIL, ONNX and rendering.

`Match` is the current complete game state: clock, phase/result, both players'
elixir, decks, hands, next cards, cycle queues, towers and the `Battle` object.
`Battle` owns units, buildings, projectiles, effects, targeting, movement,
combat and cleanup. A fixed 50 ms tick keeps positions in integer millitiles and
makes seeded runs deterministic.

## Data provenance

| Area | Source | Status |
|---|---|---|
| Unit/card stats | `tmp/gamedata`, loaded by `sim/gamedata.py` | exact to extracted client data |
| Spell/projectile stats | client data plus `data/royaleapi/combat_rules.json` overrides | exact where source is present; versioned override where documented |
| Towers | `sim/towers.py` and loaded level tables | exact for the selected data level |
| Elixir/cycle | `sim/match.py` | calibrated rule model; regeneration and cycle are explicit |
| Targeting/movement/collision order | `sim/engine.py` | approximate procedure, documented in code |
| Arena pocket/deploy geometry | `sim/arena.py` | approximate where marked unverified |
| Pixel perception and ADB | outside simulator | intentionally absent |

Unknown or unverified mechanics remain explicit in loader comments, readiness
matrices and field defaults. They are not silently filled with policy guesses.

## Interfaces

- `sim.actions.SimAction`: `WAIT`, `PLAY_CARD`, `PLAY_SPELL`, `END_STEP`.
- `sim.actions.apply_action()`: applies an action to `Match` without a device.
- `sim.diagnostics.DiagnosticSink`: deterministic compact structured event log.
- `sim.replay.Replay`: JSON replay of actions and optional diagnostic events.
- `sim.runner`: existing policy/opponent match runner.
- `scripts/benchmark_sim.py`: small sequential throughput benchmark.

## Deliberate non-goals for this phase

No new AI policy, ML training, mass dataset generation, multiprocessing or
phone integration is enabled by this layer. Search and the live bot remain
unchanged. Dataset/rollout/regret generation comes only after the microtests
and reproducibility checks pass.
