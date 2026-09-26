"""Fast Hasty-CR environment preflight.

Run before long training/evaluation so missing game data or broken imports fail
in seconds instead of after a long run.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tmp" / "gamedata" / "csv_logic"
RL = ROOT / "tmp" / "rl"


def main() -> int:
    print("Hasty-CR preflight")
    print(f"  root:   {ROOT}")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  system: {platform.system()} {platform.release()}")
    print(f"  data:   {DATA}  {'OK' if DATA.exists() else 'MISSING'}")
    print(f"  rl dir: {RL}  {'OK' if RL.exists() else 'MISSING'}")

    failures = 0
    required = ("numpy", "torch")
    for name in required:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"  import {name:<6} OK ({version})")
        except Exception as exc:
            failures += 1
            print(f"  import {name:<6} FAIL: {exc}")

    for name in ("sim.env", "sim.match", "sim.clone", "sim.train_ppo"):
        try:
            importlib.import_module(name)
            print(f"  import {name:<15} OK")
        except Exception as exc:
            failures += 1
            print(f"  import {name:<15} FAIL: {exc}")

    if not DATA.exists():
        failures += 1
        print("\nGame data is missing.")
        print("The source tree is valid, but simulator/training tests that need")
        print("csv_logic cannot run until the extracted client data is present.")

    if failures:
        print(f"\nPREFLIGHT FAILED ({failures} issue(s)).")
        return 1

    print("\nPREFLIGHT OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
