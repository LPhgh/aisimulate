#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD parity diff harness.

Diff engine behind the AFD parity gates (see the Terminal State ADR in
``docs/sweeper/afd-topology.md``; consumed by
``crates/core/parity_tests/perfmodel/test_afd_parity.py``):

- ``layers``: compare an actual layer-times payload (e.g. a Rust
  ``AfdForwardPassPerfModel.predict_afd_layer_times`` dump) against
  ``goldens/afd_layer_times.json`` at 1e-9 relative tolerance.
- ``e2e``: compare an actual ``prediction.json`` against
  ``goldens/afd_e2e.json`` at 1e-6 relative tolerance, COMMON FIELDS ONLY
  (the Rust path adds quantiles absent from the golden — expected increment).

This harness exposes ``layers`` and ``e2e`` only.

Exit code 0 = within tolerance, 1 = divergence or usage error. Every
divergent field prints golden/actual/abs-delta/rel-delta, mirroring the
report style of the engine-step parity suite.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "crates/core/parity_tests/perfmodel/goldens"

LAYER_RTOL = 1e-9
E2E_RTOL = 1e-6
# Absolute floor under a relative tolerance: golden fields that are pure
# float noise around zero (e.g. std_* of a constant frozen step, ~1e-13)
# have no meaningful relative error. 1e-9 is orders of magnitude tighter
# than the e2e rtol on any ms / tok-s quantity.
ABS_FLOOR = 1e-9

# summary fields that are counts/identifiers rather than deterministic floats
# of the measurement: compared exactly. Keys absent from the golden summary are
# not listed — a dead key here silently widens nothing but hides drift.
EXACT_KEYS = {
    "completed_requests",
    "num_requests",
    "num_e2e_latency_samples",
    "num_ttft_samples",
    "num_tpot_samples",
    "total_input_tokens",
    "total_output_tokens",
    "decode_gpus_per_worker",
    "prefill_gpus_per_worker",
}

# Wall-clock-derived fields: the replay's real execution speed on the golden
# machine, not simulated time. Excluded from the gate like every other
# wall-clock timestamp in the parity conventions.
WALL_CLOCK_KEYS = {
    "wall_time_ms",
    "processed_tokens_per_s",
    "processed_output_tokens_per_s",
}


class Divergence(Exception):
    pass


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def diff_maps(
    golden,
    actual,
    *,
    rtol: float,
    path: str = "",
    exact_keys: frozenset | set = frozenset(),
    require_keys: bool = False,
) -> list[str]:
    """Field-by-field recursive diff over golden keys; returns failure lines.

    Actual-only keys are always allowed (the Rust path legitimately adds
    fields the golden was recorded without). Golden keys missing from actual
    are only a failure when ``require_keys`` is set: the e2e gate opts in, the
    layers gate does not because four of its cases are frozen anchors that
    can never be regenerated.
    """
    failures: list[str] = []
    if isinstance(golden, dict) and isinstance(actual, dict):
        for key in sorted(golden):
            if key in WALL_CLOCK_KEYS:
                continue
            if key not in actual:
                if require_keys:
                    failures.append(f"{path}/{key}: golden key missing from actual")
                continue
            failures += diff_maps(
                golden[key],
                actual[key],
                rtol=rtol,
                path=f"{path}/{key}",
                exact_keys=exact_keys,
                require_keys=require_keys,
            )
        return failures
    if _is_number(golden) and _is_number(actual):
        leaf = path.rsplit("/", 1)[-1]
        if leaf in exact_keys:
            if golden != actual:
                failures.append(f"{path}: exact mismatch golden={golden!r} actual={actual!r}")
            return failures
        g, a = float(golden), float(actual)
        if not (math.isfinite(g) and math.isfinite(a)):
            if g != a:
                failures.append(f"{path}: non-finite mismatch golden={g!r} actual={a!r}")
            return failures
        delta = abs(g - a)
        limit = max(rtol * max(abs(g), 1e-300), ABS_FLOOR)
        if delta > limit:
            rel = delta / max(abs(g), 1e-300)
            failures.append(f"{path}: golden={g!r} actual={a!r} abs_delta={delta:.3e} rel_delta={rel:.3e} rtol={rtol}")
        return failures
    if golden != actual:
        failures.append(f"{path}: golden={golden!r} actual={actual!r}")
    return failures


def _load(path: Path):
    return json.loads(path.read_text())


def _check_layers(case: str, actual_path: Path) -> list[str]:
    golden = _load(GOLDEN_DIR / "afd_layer_times.json")["cases"][case]
    actual = _load(actual_path)
    actual_case = actual.get("cases", {}).get(case, actual)
    return diff_maps(golden, actual_case, rtol=LAYER_RTOL)


def _check_e2e(case: str, actual_path: Path) -> list[str]:
    golden_case = _load(GOLDEN_DIR / "afd_e2e.json")["cases"][case]
    prediction = _load(actual_path)
    metrics = prediction.get("summary", prediction)
    return diff_maps(
        {"summary": golden_case["summary"]},
        {"summary": metrics},
        rtol=E2E_RTOL,
        exact_keys=EXACT_KEYS,
        require_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    layers = sub.add_parser("layers", help="diff an actual layer-times JSON against the golden")
    layers.add_argument("--case", required=True, help="case id in goldens/afd_layer_times.json")
    layers.add_argument("--actual", required=True, type=Path, help="actual layer-times JSON (full fixture or one case)")

    e2e = sub.add_parser("e2e", help="diff an actual prediction.json against the e2e golden")
    e2e.add_argument("--case", required=True, help="case id in goldens/afd_e2e.json")
    e2e.add_argument("--actual", required=True, type=Path, help="actual prediction.json")

    args = parser.parse_args(argv)
    try:
        if args.command == "layers":
            failures = _check_layers(args.case, args.actual)
        else:
            failures = _check_e2e(args.case, args.actual)
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"parity harness error: {exc}", file=sys.stderr)
        return 1

    if failures:
        print(f"PARITY DIVERGED ({len(failures)} fields):", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("parity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
