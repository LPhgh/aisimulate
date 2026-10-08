# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real SDK estimate -> frozen target assertions for analytical AFD."""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from pathlib import Path

import pytest

import aisimulate_core
from aisimulate.legacy_cli.api import cli_estimate

pytestmark = [pytest.mark.integration, pytest.mark.gpu_0, pytest.mark.pre_merge]

_BASELINE = json.loads((Path(__file__).with_name("afd_estimate_baseline.json")).read_text())
_EXPECTED_CASE_IDS = {
    "decode-conservative",
    "decode-optimistic",
    "decode-serial",
    "uneven-microbatch",
    "optimistic-fallback",
    "prefill",
    "both",
    "invalid-pipeline",
    "invalid-combined-both",
    "microbatch-one",
    "microbatch-four",
    "attention-tp-two",
    "attention-tp-four",
    "ffn-ep-four",
    "attention-two-nodes",
    "ffn-two-nodes",
    "short-context",
    "long-context",
    "communication-half",
    "communication-one",
    "boundary-on-ffn",
    "decode-with-pd",
    "prefill-with-pd",
    "invalid-phase",
    "invalid-missing-attention-nodes",
    "invalid-missing-ffn-nodes",
    "invalid-zero-microbatches",
    "invalid-zero-attention-tp",
    "invalid-ffn-ep-not-divisor",
    "invalid-zero-attention-nodes",
    "invalid-zero-a-batch-size",
    "invalid-zero-comm-overhead",
    "invalid-attention-tp-not-divisor",
    "invalid-oom",
}
# Rejections the manifest is allowed to pin. Keeping this explicit means an unusable ``type``
# fails the manifest integrity test instead of surfacing as a KeyError mid-case.
_ERROR_TYPES = {"ValueError": ValueError, "NotImplementedError": NotImplementedError, "RuntimeError": RuntimeError}
# Overrides that only the static complement of a combined AFD+PD query consumes, and that the
# common query therefore does not list.
_STATIC_COMPLEMENT_OVERRIDE_KEYS = {"tp_size", "moe_tp_size", "moe_ep_size"}


def test_baseline_manifest_is_complete():
    cases = _BASELINE["cases"]
    assert len(cases) == len(_EXPECTED_CASE_IDS)
    assert {case["id"] for case in cases} == _EXPECTED_CASE_IDS
    allowed_overrides = set(_BASELINE["query"]) | _STATIC_COMPLEMENT_OVERRIDE_KEYS
    for case in cases:
        # A typo'd override key would otherwise reach cli_estimate and fail as a distant TypeError.
        assert isinstance(case["overrides"], dict), case["id"]
        unexpected = sorted(set(case["overrides"]) - allowed_overrides)
        assert not unexpected, (case["id"], unexpected)
        if case["id"].startswith("invalid-"):
            assert set(case["error"]) == {"type", "match", "origin"}
            assert all(case["error"].values())
            assert case["error"]["type"] in _ERROR_TYPES, case["id"]
            assert "metrics" not in case
        else:
            assert "error" not in case
            assert {"metrics", "path", "phases"} <= case.keys()
            query = {**_BASELINE["query"], **case["overrides"]}
            phase = query["afd_phase"]
            expected = {"prefill": {"ttft"}, "decode": {"tpot"}, "both": {"ttft", "tpot"}}[phase]
            if query["afd_combined_with_pd"]:
                expected = {"ttft", "tpot"}
            assert set(case["metrics"]) == expected
            assert all(math.isfinite(value) and value > 0 for value in case["metrics"].values())
            assert set(case["phases"]) == ({"prefill", "decode"} if phase == "both" else {phase})


@pytest.fixture(scope="module")
def frozen_inputs():
    """Fail closed if model or performance data no longer matches the baseline."""
    root = Path(aisimulate_core.__file__).parent
    data_files = {str(path.relative_to(root)) for path in (root / "systems/data/h200_sxm").rglob("*") if path.is_file()}
    expected_data_files = {path for path in _BASELINE["input_sha256"] if path.startswith("systems/data/h200_sxm/")}
    assert data_files == expected_data_files, "H200 database file inventory changed"
    for relative, expected in _BASELINE["input_sha256"].items():
        path = root / relative
        assert path.is_file(), f"Missing baseline input: {path}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"Baseline input changed: {relative}; review and recapture explicitly"


@pytest.mark.parametrize("case", _BASELINE["cases"], ids=lambda case: case["id"])
def test_afd_estimate_matches_frozen_target(case, frozen_inputs, tmp_path, caplog):
    query = {**_BASELINE["query"], **case["overrides"]}
    evidence = {
        "id": case["id"],
        "query": query,
        "baseline_commit": case.get("target_commit", _BASELINE["commit"]),
        "input_commit": _BASELINE["input_commit"],
    }
    try:
        if "error" in case:
            expected = case["error"]
            with pytest.raises(_ERROR_TYPES[expected["type"]], match=expected["match"]) as caught:
                cli_estimate(**query)
            # Pin the concrete class: pytest.raises accepts subclasses too, and AFDPartitionError is
            # a ValueError that must not silently satisfy a plain-ValueError expectation.
            assert type(caught.value).__name__ == expected["type"]
            origin = traceback.extract_tb(caught.value.__traceback__)[-1]
            evidence["error"] = {
                "type": type(caught.value).__name__,
                "message": str(caught.value),
                "origin": origin.name,
            }
            assert origin.name == expected["origin"]
            return

        result = cli_estimate(**query)
        expected_metrics = {"prefill": {"ttft"}, "decode": {"tpot"}, "both": {"ttft", "tpot"}}
        required = {"ttft", "tpot"} if query["afd_combined_with_pd"] else expected_metrics[query["afd_phase"]]
        assert set(case["metrics"]) == required
        evidence["metrics"] = {}
        for metric, target in case["metrics"].items():
            actual = getattr(result, metric)
            evidence["metrics"][metric] = {
                "target": target,
                "actual": actual,
                "delta": actual - target,
                "relative_delta": (actual - target) / target if target else None,
            }
        evidence["path"] = {key: result.raw[key] for key in case["path"]}
        assert result.mode == "afd"
        assert result.backend_version == query["backend_version"]
        for metric, values in evidence["metrics"].items():
            actual, target = values["actual"], values["target"]
            assert math.isfinite(actual) and actual > 0, (case["id"], metric, values)
            # Public SDK latencies are already rounded to milliseconds at 3 decimals.
            # Compare that contract exactly; never regenerate targets during a test.
            assert actual == target, (case["id"], metric, values)
        assert evidence["path"] == case["path"]
        assert set(result.raw["afd_layer_measurements"]) == set(case["phases"])
        if case["id"] == "optimistic-fallback":
            assert "Falling back to conservative model" in caplog.text
    except Exception as exc:
        evidence["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        evidence["logs"] = caplog.text
        (tmp_path / f"{case['id']}.json").write_text(json.dumps(evidence, indent=2) + "\n")


@pytest.mark.xfail(
    strict=True,
    reason="AFD+MTP silently drops afd_layer_measurements: the decode/static projection rebuilds the "
    "result dict from a ColumnsAFD frame that omits the key. See tests/test_afd_projection_contract.py.",
)
def test_afd_mtp_preserves_layer_measurements(frozen_inputs):
    """Strict xfail, so an XPASS means the defect is fixed: retire this test and the canary together.

    Kept out of the frozen matrix on purpose. It pins no latency target, and adding it as a case
    would bless the current lossy behaviour as expected.
    """
    query = {**_BASELINE["query"], "afd_phase": "decode", "nextn": 1, "nextn_accepted": 1.0}
    result = cli_estimate(**query)
    assert "afd_layer_measurements" in result.raw
