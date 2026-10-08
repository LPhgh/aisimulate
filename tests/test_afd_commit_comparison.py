# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The paired gate must reject drift even when both executions finish."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "compare_afd_commits", Path(__file__).resolve().parents[1] / "scripts/compare_afd_commits.py"
)
_HARNESS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HARNESS)


@pytest.fixture
def pair():
    query = {"afd_phase": "decode", "backend_version": "test"}
    case = {
        "id": "decode",
        "overrides": {},
        "metrics": {"tpot": 10.0},
        "path": {"phase": "decode"},
        "phases": ["decode"],
    }
    manifest = {"query": query, "input_sha256": {"input": "hash"}, "cases": [case]}
    row = {
        "id": "decode",
        "query": query,
        "metrics": {"tpot": 10.0},
        "path": case["path"],
        "phases": ["decode"],
        "mode": "afd",
        "backend_version": "test",
    }
    snapshot = {"input_sha256": {"input": "hash"}, "cases": [row]}
    return deepcopy(snapshot), deepcopy(snapshot), deepcopy(manifest)


def test_identical_pair_passes(pair):
    result = _HARNESS.compare(*pair, 0)
    assert result["passed"]
    assert result["deltas"][0]["delta_ms"] == 0


@pytest.mark.parametrize(
    "mutation", ["slower", "faster", "missing", "duplicate", "empty-metrics", "nan", "path", "inputs", "query", "error"]
)
def test_gate_rejects_corruption(pair, mutation):
    before, after, manifest = pair
    row = after["cases"][0]
    if mutation in ("slower", "faster", "nan"):
        row["metrics"]["tpot"] = {"slower": 11.0, "faster": 9.0, "nan": float("nan")}[mutation]
    elif mutation == "missing":
        after["cases"] = []
    elif mutation == "duplicate":
        after["cases"].append(deepcopy(row))
    elif mutation == "empty-metrics":
        row["metrics"] = {}
    elif mutation == "path":
        row["path"]["phase"] = "prefill"
    elif mutation == "inputs":
        after["input_sha256"]["input"] = "changed"
    elif mutation == "query":
        # Matching but incorrect queries on BOTH sides must also fail.
        before["cases"][0]["query"]["extra"] = True
        row["query"]["extra"] = True
    elif mutation == "error":
        row["error"] = {"type": "RuntimeError", "message": "unexpected", "origin": "run"}
    assert not _HARNESS.compare(before, after, manifest, 0)["passed"]


def test_input_drift_requires_explicit_mode(pair):
    before, after, manifest = pair
    after["input_sha256"] = {"input": "changed"}
    result = _HARNESS.compare(before, after, manifest, 0, allow_input_drift=True)
    assert result["passed"]
    assert result["comparison_kind"] == "code-and-data"


def test_expected_errors_are_checked_on_both_sides(pair):
    before, after, manifest = pair
    manifest["cases"][0] = {
        "id": "decode",
        "overrides": {},
        "error": {"type": "ValueError", "match": "invalid phase", "origin": "cli_estimate"},
    }
    error = {"type": "ValueError", "message": "invalid phase: bad", "origin": "cli_estimate"}
    for snapshot in (before, after):
        snapshot["cases"][0] = {"id": "decode", "query": manifest["query"], "error": deepcopy(error)}
    assert _HARNESS.compare(before, after, manifest, 0)["passed"]
    for snapshot in (before, after):
        snapshot["cases"][0]["error"]["origin"] = "wrong_layer"
    assert not _HARNESS.compare(before, after, manifest, 0)["passed"]


def test_empty_manifest_is_not_a_pass(pair):
    before, after, manifest = pair
    manifest["cases"] = []
    before["cases"] = after["cases"] = []
    with pytest.raises(ValueError, match="must be nonempty with unique IDs"):
        _HARNESS.compare(before, after, manifest, 0)


@pytest.mark.parametrize("corruption", ["no-path", "no-error-origin"])
def test_malformed_manifest_case_is_rejected(pair, corruption):
    before, after, manifest = pair
    case = manifest["cases"][0]
    if corruption == "no-path":
        del case["path"]
    else:
        case["error"] = {"type": "ValueError", "match": "invalid phase"}
    with pytest.raises(ValueError, match="decode: manifest case missing"):
        _HARNESS.compare(before, after, manifest, 0)


def test_pd_combination_requires_complementary_metric(pair):
    before, after, manifest = pair
    manifest["query"]["afd_combined_with_pd"] = True
    for snapshot in (before, after):
        snapshot["cases"][0]["query"]["afd_combined_with_pd"] = True
    assert not _HARNESS.compare(before, after, manifest, 0)["passed"]


def test_sides_agreeing_on_a_wrong_database_still_fail(pair):
    """The paired gate alone has no absolute anchor: both sides must match the manifest baseline."""
    before, after, manifest = pair
    for snapshot in (before, after):
        snapshot["input_sha256"] = {"input": "some-other-database"}
    result = _HARNESS.compare(before, after, manifest, 0)
    assert result["inputs_equal"], "sides agree with each other, so only the manifest gate can catch this"
    assert not result["inputs_match_manifest"]
    assert not result["passed"]
    assert len([item for item in result["failures"] if "inputs differ from manifest baseline" in item]) == 2


def test_missing_manifest_input_is_detected(pair):
    """A file absent from disk is invisible to a side-by-side hash comparison of what remains."""
    before, after, manifest = pair
    manifest["input_sha256"] = {"input": "hash", "deleted": "hash"}
    result = _HARNESS.compare(before, after, manifest, 0)
    assert not result["passed"]
    assert result["input_drift"]["before"]["missing"] == ["deleted"]
    assert result["input_drift"]["after"]["missing"] == ["deleted"]


def test_input_drift_lists_differing_keys(pair):
    before, after, manifest = pair
    after["input_sha256"] = {"input": "changed", "untracked": "hash"}
    result = _HARNESS.compare(before, after, manifest, 0, allow_input_drift=True)
    assert result["passed"]
    assert result["comparison_kind"] == "code-and-data"
    assert not result["inputs_match_manifest"], "drift is tolerated, never hidden"
    assert result["input_drift"]["after"]["changed"] == ["input"]
    assert result["input_drift"]["after"]["extra"] == ["untracked"]
    assert result["input_drift"]["before"] == {"missing": [], "extra": [], "changed": []}


def test_check_targets_gates_a_drift_both_sides_share(pair):
    before, after, manifest = pair
    for snapshot in (before, after):
        snapshot["cases"][0]["metrics"]["tpot"] = 11.0
    assert _HARNESS.compare(before, after, manifest, 0)["passed"], "paired-only gate cannot see a shared drift"
    result = _HARNESS.compare(before, after, manifest, 0, check_targets=True)
    assert not result["passed"]
    assert len([item for item in result["failures"] if "deviation from frozen target" in item]) == 2


def test_target_deviations_are_reported_without_gating(pair):
    before, after, manifest = pair
    after["cases"][0]["metrics"]["tpot"] = 11.0
    result = _HARNESS.compare(before, after, manifest, 0.2)
    assert result["passed"]
    assert result["check_targets"] is False
    deviations = {(row["side"], row["metric"]): row["relative_deviation"] for row in result["target_deviations"]}
    assert deviations[("before", "tpot")] == 0.0
    assert deviations[("after", "tpot")] == pytest.approx(0.1)


def test_structural_failure_returns_instead_of_raising(pair):
    """A manifest/snapshot length mismatch must report, not escape the strict zip as ValueError."""
    before, after, manifest = pair
    manifest["cases"].append({**deepcopy(manifest["cases"][0]), "id": "second"})
    result = _HARNESS.compare(before, after, manifest, 0)
    assert not result["passed"]
    assert result["deltas"] == []
    assert result["target_deviations"] == []
    assert any("missing, duplicate, reordered or unexpected cases" in item for item in result["failures"])


def test_environment_diff_is_reported_but_not_gated(pair):
    before, after, manifest = pair
    before.update(packages={"shared": "1", "updated": "1", "dropped": "1"}, python="3.13.7", executable="/x/python")
    after.update(packages={"shared": "1", "updated": "2", "added": "3"}, python="3.13.8", executable="/y/python")
    result = _HARNESS.compare(before, after, manifest, 0)
    assert result["passed"], "dependency equality is recorded for attribution, never enforced"
    environment = result["environment_diff"]
    assert environment["added"] == {"added": "3"}
    assert environment["removed"] == {"dropped": "1"}
    assert environment["changed"] == {"updated": {"before": "1", "after": "2"}}
    assert environment["python"] == {"before": "3.13.7", "after": "3.13.8"}
    assert environment["executable"] == {"before": "/x/python", "after": "/y/python"}


@pytest.mark.parametrize("corruption", ["no-query", "no-input-fingerprints", "no-overrides"])
def test_malformed_manifest_header_is_rejected(pair, corruption):
    before, after, manifest = pair
    if corruption == "no-query":
        del manifest["query"]
    elif corruption == "no-input-fingerprints":
        del manifest["input_sha256"]
    else:
        del manifest["cases"][0]["overrides"]
    with pytest.raises(ValueError, match="manifest"):
        _HARNESS.compare(before, after, manifest, 0)


def test_comparison_kind_is_paired_only_with_baseline_inputs(pair):
    """Agreeing with each other is not 'paired code' when the shared inputs are not the baseline's."""
    before, after, manifest = pair
    assert _HARNESS.compare(before, after, manifest, 0)["comparison_kind"] == "paired-code"
    for snapshot in (before, after):
        snapshot["input_sha256"] = {"input": "some-other-database"}
    for allow in (False, True):
        result = _HARNESS.compare(before, after, manifest, 0, allow_input_drift=allow)
        assert result["inputs_equal"], "the two sides still agree with each other"
        assert not result["inputs_match_manifest"]
        assert result["comparison_kind"] == "code-and-data", "the label must not advertise isolation it lacks"


def test_render_report_exposes_anchor_drift_and_environment(pair):
    """report.md is the operator-facing artifact; main() has no other coverage."""
    before, after, manifest = pair
    before.update(packages={"pkg": "1"}, python="3.13.7", executable="/x/python")
    after.update(packages={"pkg": "2"}, python="3.13.7", executable="/x/python")
    after["cases"][0]["metrics"]["tpot"] = 11.0
    result = _HARNESS.compare(before, after, manifest, 0.2)
    result.update(before_commit="a" * 40, after_commit="b" * 40, case_count=1, relative_tolerance=0.2)
    report = _HARNESS.render_report(result)
    assert "# AFD commit comparison: PASS" in report
    assert "Absolute anchor: none, paired sides only" in report
    assert "Inputs match manifest baseline: True" in report
    assert "| Before vs Target % | After vs Target % |" in report
    assert "| decode | tpot | 10.0 | 11.0 | 1.000000 | 10.000000 | 0.000000 | 10.000000 |" in report
    assert "- changed `pkg` 1 -> 2" in report
    assert "## Failures" in report and "- none" in report
    assert "## Input fingerprint drift" not in report


def test_render_report_lists_drift_and_failures(pair):
    before, after, manifest = pair
    for snapshot in (before, after):
        snapshot["input_sha256"] = {"input": "some-other-database"}
    result = _HARNESS.compare(before, after, manifest, 0)
    result.update(before_commit="a" * 40, after_commit="b" * 40, case_count=1, relative_tolerance=0.0)
    report = _HARNESS.render_report(result)
    assert "# AFD commit comparison: FAIL" in report
    assert "Inputs match manifest baseline: False" in report
    assert "## Input fingerprint drift" in report
    assert "`input`" in report
    assert "inputs differ from manifest baseline" in report
