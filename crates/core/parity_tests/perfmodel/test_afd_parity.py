# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-class AFD parity suite (continuous-batching rebuild).

Pins the continuous AFD replay path (SglangCore + injected ``aic-afd``
timing provider) against the rebuilt goldens:

- `TestAfdLayerTimesGolden` — the compiled per-pool EngineSpec fixtures
  (``goldens/afd_specs/``) stay loadable and carry the measurement identity
  the Rust `AfdForwardPassPerfModel` parity tests compare against (the
  bit-level 1e-9 assertions live in `cargo test perfmodel::afd`; this class
  anchors the fixtures + golden schema from the Python side). The four
  trtllm cases are the FROZEN anchor, produced at ``9cd9e8e9^`` and
  non-regenerable; the sglang case is self-anchored by the native model as a
  data-drift guard for the perf-DB family the continuous path consumes.
- `TestAfdE2eGolden` — `AISIMULATE_AFD_NATIVE=1` runs of the committed demo
  configs reproduce ``goldens/afd_e2e.json`` within 1e-6 relative (common
  fields, wall-clock-derived fields excluded). Two cases bracket the admission
  boundary: the unsaturated demo config and a saturated variant whose
  concurrency exceeds `max_num_seqs`. The retired batch-locked AfdCore golden
  and the two afd+pd cases were dropped: the two-pool handoff is fail-loud
  future work on the continuous path.

Run from the repository root:

    python/aisimulate/.venv/bin/pytest -q crates/core/parity_tests/perfmodel/test_afd_parity.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "python/aisimulate/src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
AISIMULATE_CLI = _REPO_ROOT / "python/aisimulate/.venv/bin/aisimulate"

import afd_parity

E2E_CASES = (
    "qwen3-32b-h200-sglang-pure-both",
    "qwen3-32b-h200-sglang-pure-both-saturated",
    "qwen3-32b-h200-sglang-fill-amortized",
)


def _golden(filename: str) -> dict:
    return json.loads((GOLDEN_DIR / filename).read_text())


def _run_native(config: Path) -> dict:
    env = dict(os.environ)
    env["AISIMULATE_AFD_NATIVE"] = "1"
    with tempfile.TemporaryDirectory(prefix="afd-parity-") as run_dir:
        subprocess.run(
            [str(AISIMULATE_CLI), "predict", "--config", str(config), "--output-dir", run_dir, "--format", "json"],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        return json.loads((Path(run_dir) / "prediction.json").read_text())


class TestAfdLayerTimesGolden:
    def test_spec_fixtures_cover_every_homo_and_hetero_pool(self):
        fixtures = {path.name for path in (GOLDEN_DIR / "afd_specs").glob("*.json")}
        assert {"pure_h200_a.json", "pure_h200_f.json", "hetero_b200_f.json"} <= fixtures

    def test_golden_cases_cover_pure_pd_and_hetero(self):
        cases = _golden("afd_layer_times.json")["cases"]
        assert {
            "qwen3-32b-h200-trtllm-pure-both",
            "qwen3-32b-h200-trtllm-pd-decode",
            "qwen3-32b-h200-trtllm-pd-prefill",
            "qwen3-32b-h200a-b200f-trtllm-hetero-both",
            "qwen3-32b-h200-sglang-pure-both",
        } <= set(cases)
        pure = cases["qwen3-32b-h200-trtllm-pure-both"]
        assert set(pure["phases"]) == {"prefill", "decode"}
        for phase in pure["phases"].values():
            for field in ("attention_ms", "ffn_ms", "a_to_f_ms", "f_to_a_ms"):
                assert math.isfinite(phase[field]) and phase[field] > 0.0
            assert phase["num_layers"] == 64
        hetero = cases["qwen3-32b-h200a-b200f-trtllm-hetero-both"]
        assert hetero["identity"]["a_system"] == "h200_sxm"
        assert hetero["identity"]["f_system"] == "b200_sxm"
        # The hetero composition pins per-pool reads: attention from the A
        # pool's h200 DB (identical to the homo case), ffn from b200.
        homo_phases = pure["phases"]
        for phase_name in ("prefill", "decode"):
            assert hetero["phases"][phase_name]["attention_ms"] == homo_phases[phase_name]["attention_ms"]
            assert hetero["phases"][phase_name]["ffn_ms"] != homo_phases[phase_name]["ffn_ms"]

    def test_sglang_case_anchors_the_continuous_path_db_family(self):
        case = _golden("afd_layer_times.json")["cases"]["qwen3-32b-h200-sglang-pure-both"]
        identity = case["identity"]
        assert identity["backend"] == "sglang"
        assert identity["topology"]["phase"] == "both"
        assert identity["topology"]["combined_with_pd"] is False
        assert set(case["phases"]) == {"prefill", "decode"}
        for phase in case["phases"].values():
            for field in ("attention_ms", "ffn_ms", "a_to_f_ms", "f_to_a_ms"):
                assert math.isfinite(phase[field]) and phase[field] > 0.0


@pytest.mark.parametrize("case_id", E2E_CASES)
class TestAfdE2eGolden:
    def test_native_run_matches_golden(self, case_id: str):
        config = _REPO_ROOT / _golden("afd_e2e.json")["cases"][case_id]["config"]
        # The demo configs are committed parity assets: a missing file means
        # the repository is broken, not that the case is optional.
        assert config.is_file(), f"golden demo config missing: {config}"
        prediction = _run_native(config)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(prediction, handle)
            actual_path = Path(handle.name)
        failures = afd_parity._check_e2e(case_id, actual_path)
        assert not failures, "AFD native e2e diverged from the golden:\n" + "\n".join(failures)
