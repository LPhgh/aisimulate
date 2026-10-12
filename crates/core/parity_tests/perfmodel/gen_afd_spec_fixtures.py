#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the AFD per-pool EngineSpec fixtures under ``goldens/afd_specs/``.

Each fixture is one AFD pool's compiled op-list spec (schema 22), produced by
the standard ``compile_engine`` path at the pool identity the Rust
``pool_fpm_config`` derives (``crates/core/src/perfmodel/afd/model.rs``):

- A pool: ``tp = tp_a = 8``, ``moe_tp = 8``, ``moe_ep = 1``;
- F pool: ``tp = n_f_nodes * gpus_per_node = 8``, ``moe_tp = tp / ep``,
  ``moe_ep = 1``; the hetero variant compiles the same shape against
  ``b200_sxm``.

The Rust parity tests (``perfmodel::afd::model`` and
``engine::scheduler::afd::timing``) load these fixtures and compare against
``goldens/afd_layer_times.json`` — layer-time goldens produced at commit
``9cd9e8e9^`` and frozen: the code that produced them is gone, so THOSE
numbers cannot be regenerated (these spec fixtures can).

Regenerate (from the repository root):

    python/aisimulate/.venv/bin/python crates/core/parity_tests/perfmodel/gen_afd_spec_fixtures.py

Compilation is deterministic for a given perf-data tree, so with unchanged
data the run verifies the committed fixtures byte-for-byte and exits 0;
any drift is reported and rewritten (exit 1 signals that the golden Rust
tests must be re-examined before committing the new fixtures).
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python/aisimulate/src"))

import aisimulate_core
from aisimulate_core.sdk.engine import compile_engine

OUT_DIR = os.path.join(HERE, "goldens", "afd_specs")

MODEL = "Qwen/Qwen3-32B"
BACKEND = "trtllm"
VERSION = "1.3.0rc20"

# fixture name -> compile identity (system + pool parallel shape).
FIXTURES = {
    "pure_h200_a.json": {"system": "h200_sxm", "tp_size": 8, "moe_tp_size": 8, "moe_ep_size": 1},
    "pure_h200_f.json": {"system": "h200_sxm", "tp_size": 8, "moe_tp_size": 8, "moe_ep_size": 1},
    "hetero_b200_f.json": {"system": "b200_sxm", "tp_size": 8, "moe_tp_size": 8, "moe_ep_size": 1},
}

# Continuous-path anchors: the SglangCore-injected AFD runtime queries the
# sglang perf-DB family, so the layer-time golden's sglang case needs spec
# fixtures compiled at the sglang identity.
SGLANG_FIXTURES = {
    "pure_h200_a_sglang.json": {"system": "h200_sxm", "tp_size": 8, "moe_tp_size": 8, "moe_ep_size": 1},
    "pure_h200_f_sglang.json": {"system": "h200_sxm", "tp_size": 8, "moe_tp_size": 8, "moe_ep_size": 1},
}
SGLANG_BACKEND = "sglang"
SGLANG_VERSION = "0.5.14"


def _capture_spec_json(identity: dict, backend: str, version: str) -> str:
    """Run compile_engine and capture its intermediate EngineSpec JSON.

    compile_engine returns bincode; the fixtures are the JSON form. The
    bincode converter is the single call site that sees the spec JSON, so
    the capture spies on it instead of duplicating compile_engine's body.
    """
    captured: dict[str, str] = {}
    original = aisimulate_core.engine_spec_bincode_from_json

    def spy(spec_json):
        captured["json"] = spec_json
        return original(spec_json)

    aisimulate_core.engine_spec_bincode_from_json = spy
    try:
        compile_engine(
            MODEL,
            identity["system"],
            backend,
            version,
            tp_size=identity["tp_size"],
            moe_tp_size=identity["moe_tp_size"],
            moe_ep_size=identity["moe_ep_size"],
        )
    finally:
        aisimulate_core.engine_spec_bincode_from_json = original
    return captured["json"]


def main() -> int:
    drifted = []
    targets = [(name, identity, BACKEND, VERSION) for name, identity in FIXTURES.items()] + [
        (name, identity, SGLANG_BACKEND, SGLANG_VERSION) for name, identity in SGLANG_FIXTURES.items()
    ]
    for name, identity, backend, version in targets:
        spec = json.loads(_capture_spec_json(identity, backend, version))
        rendered = json.dumps(spec, indent=2, sort_keys=True) + "\n"
        path = os.path.join(OUT_DIR, name)
        previous = None
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                previous = handle.read()
        if previous == rendered:
            print(f"{name}: unchanged")
            continue
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        drifted.append(name)
        print(f"{name}: {'written (no committed fixture)' if previous is None else 'DRIFTED — rewritten'}")
    if drifted:
        print(
            f"spec fixtures changed: {drifted} — re-run the Rust golden tests "
            "(cargo test perfmodel::afd, scheduler::afd) and re-examine the "
            "frozen layer-time goldens before committing.",
            file=sys.stderr,
        )
        return 1
    print("all fixtures reproduce byte-for-byte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
