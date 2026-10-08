# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the same AFD queries in two prepared checkouts and fail on drift.

Each checkout needs its own `uv sync --project python/aisimulate --extra dev`.
This harness never checks out revisions, installs dependencies or rewrites targets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def validate_manifest(manifest):
    if not isinstance(manifest.get("query"), dict) or not manifest["query"]:
        raise ValueError("Case manifest must define a nonempty common 'query' object")
    if not isinstance(manifest.get("input_sha256"), dict) or not manifest["input_sha256"]:
        raise ValueError("Case manifest must define a nonempty 'input_sha256' fingerprint map")
    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Case manifest must be nonempty with unique IDs")
    for case in cases:
        if not isinstance(case.get("overrides"), dict):
            raise ValueError(f"{case['id']}: manifest case missing ['overrides']")
        if "error" in case:
            required = {"type", "match", "origin"} - case["error"].keys()
        else:
            # ``metrics`` has two roles: the frozen golden value asserted by the pytest suite, and
            # the optional absolute anchor for --check-targets. Without that flag ``compare`` only
            # pairs the two sides against each other.
            required = {"metrics", "path", "phases"} - case.keys()
        if required:
            raise ValueError(f"{case['id']}: manifest case missing {sorted(required)}")


def input_drift(snapshot, manifest):
    """Diff one snapshot's fingerprints against the manifest baseline.

    Catching files that vanished from disk matters as much as changed hashes: a snapshot rebuilt
    by walking the systems directory simply omits absent files, so both sides can agree on a
    database that no longer matches the baseline.
    """
    expected = manifest["input_sha256"]
    actual = snapshot.get("input_sha256", {})
    return {
        "missing": sorted(set(expected) - set(actual)),
        "extra": sorted(set(actual) - set(expected)),
        "changed": sorted(key for key in set(expected) & set(actual) if expected[key] != actual[key]),
    }


def environment_diff(before, after):
    """Reported for attribution only; dependency equality is deliberately not gated."""
    left, right = before.get("packages", {}), after.get("packages", {})
    diff = {
        "added": {name: right[name] for name in sorted(set(right) - set(left))},
        "removed": {name: left[name] for name in sorted(set(left) - set(right))},
        "changed": {
            name: {"before": left[name], "after": right[name]}
            for name in sorted(set(left) & set(right))
            if left[name] != right[name]
        },
    }
    for key in ("python", "executable"):
        if before.get(key) != after.get(key):
            diff[key] = {"before": before.get(key), "after": after.get(key)}
    return diff


def capture(repo, manifest, systems_path):
    validate_manifest(manifest)
    # Select by files, not ImportError: a broken dependency must fail visibly.
    source = repo / "python/aisimulate/src"
    modern = (source / "aisimulate/legacy_cli/api.py").is_file()
    core = importlib.import_module("aisimulate_core" if modern else "aiconfigurator_core")
    api = importlib.import_module("aisimulate.legacy_cli.api" if modern else "aiconfigurator.cli.api")
    for module in (core, api):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError(f"Wrong checkout imported: {module.__file__}")
    dirty = git(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "python/aisimulate",
        "crates",
        "Cargo.toml",
        "Cargo.lock",
    )
    if dirty:
        raise RuntimeError(f"Runtime checkout must be clean: {dirty}")
    root = Path(core.__file__).parent
    system_root = systems_path or root / "systems"
    inputs = {}
    for relative in manifest["input_sha256"]:
        if not relative.startswith("systems/"):
            inputs[relative] = digest(root / relative)
    inputs["systems/h200_sxm.yaml"] = digest(system_root / "h200_sxm.yaml")
    for path in sorted((system_root / "data/h200_sxm").rglob("*")):
        if path.is_file():
            inputs["systems/" + str(path.relative_to(system_root))] = digest(path)
    rows = []
    for case in manifest["cases"]:
        query = {**manifest["query"], **case["overrides"]}
        if systems_path:
            query["systems_paths"] = str(systems_path)
        row = {"id": case["id"], "query": query}
        try:
            result = api.cli_estimate(**query)
            phase = query["afd_phase"]
            keys = {"prefill": ["ttft"], "decode": ["tpot"], "both": ["ttft", "tpot"]}[phase]
            if query.get("afd_combined_with_pd"):
                keys = ["ttft", "tpot"]
            row.update(
                metrics={key: getattr(result, key) for key in keys},
                path={key: result.raw[key] for key in case.get("path", {})},
                phases=sorted(result.raw["afd_layer_measurements"]),
                mode=result.mode,
                backend_version=result.backend_version,
            )
        except Exception as exc:
            row["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "origin": traceback.extract_tb(exc.__traceback__)[-1].name,
            }
        rows.append(row)
    return {
        "commit": git(repo, "rev-parse", "HEAD"),
        "repo": str(repo),
        "python": sys.version,
        "executable": sys.executable,
        "api": api.__file__,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "input_sha256": inputs,
        "cases": rows,
    }


def compare(before, after, manifest, tolerance, allow_input_drift=False, check_targets=False):
    validate_manifest(manifest)
    failures = []
    deltas = []
    target_deviations = []
    expected_ids = [case["id"] for case in manifest["cases"]]
    inputs_equal = before["input_sha256"] == after["input_sha256"]
    if not inputs_equal and not allow_input_drift:
        failures.append("Input fingerprints differ: use common --systems-path for paired code regression")
    drift = {label: input_drift(snapshot, manifest) for label, snapshot in (("before", before), ("after", after))}
    inputs_match_manifest = not any(any(delta.values()) for delta in drift.values())
    if not inputs_match_manifest and not allow_input_drift:
        for label, delta in drift.items():
            if any(delta.values()):
                failures.append(
                    f"{label}: inputs differ from manifest baseline ({len(delta['missing'])} missing, "
                    f"{len(delta['extra'])} extra, {len(delta['changed'])} changed)"
                )
    structural_failure = False
    for label, snapshot in (("before", before), ("after", after)):
        if [row["id"] for row in snapshot["cases"]] != expected_ids:
            failures.append(f"{label}: missing, duplicate, reordered or unexpected cases")
            structural_failure = True
    common = {
        "inputs_equal": inputs_equal,
        "inputs_match_manifest": inputs_match_manifest,
        "input_drift": drift,
        # Both gates define "paired code": agreeing with each other is not enough if the shared
        # inputs are not the baseline's, otherwise the label advertises an isolation it lacks.
        "comparison_kind": "paired-code" if inputs_equal and inputs_match_manifest else "code-and-data",
        "check_targets": check_targets,
        "environment_diff": environment_diff(before, after),
    }
    # An explicit flag, not a substring of the failure text: the zip below is strict, so a case
    # count mismatch has to short-circuit or it raises instead of reporting.
    if structural_failure:
        return {"passed": False, "failures": failures, "deltas": [], "target_deviations": [], **common}
    for case, old, new in zip(manifest["cases"], before["cases"], after["cases"], strict=True):
        name = case["id"]
        expected_query = {**manifest["query"], **case["overrides"]}
        for label, row in (("before", old), ("after", new)):
            if {k: v for k, v in row["query"].items() if k != "systems_paths"} != expected_query:
                failures.append(f"{name}/{label}: query differs from manifest")
        if old["query"] != new["query"]:
            failures.append(f"{name}: resolved queries differ")
        if "error" in case:
            target = case["error"]
            for label, row in (("before", old), ("after", new)):
                error = row.get("error", {})
                if (
                    error.get("type") != target["type"]
                    or error.get("origin") != target["origin"]
                    or not re.search(target["match"], error.get("message", ""))
                ):
                    failures.append(f"{name}/{label}: unexpected rejection {error}")
            continue
        if "error" in old or "error" in new:
            failures.append(f"{name}: unexpected execution error")
            continue
        phase = {**manifest["query"], **case["overrides"]}["afd_phase"]
        required = {"decode": {"tpot"}, "prefill": {"ttft"}, "both": {"ttft", "tpot"}}[phase]
        if expected_query.get("afd_combined_with_pd"):
            required = {"ttft", "tpot"}
        for label, row in (("before", old), ("after", new)):
            if set(row["metrics"]) != required:
                failures.append(f"{name}/{label}: wrong metric coverage")
            if row["path"] != case["path"] or row["phases"] != sorted(case["phases"]):
                failures.append(f"{name}/{label}: path or phase contract changed")
            if row["mode"] != "afd" or row["backend_version"] != row["query"]["backend_version"]:
                failures.append(f"{name}/{label}: wrong mode or backend version")
        for metric in required:
            left, right = old["metrics"].get(metric), new["metrics"].get(metric)
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (left, right)):
                failures.append(f"{name}/{metric}: missing or invalid latency")
                continue
            # Single-sided relative change, so a 0.01 bound admits +1% but also -0.99%.
            relative = (right - left) / left
            deltas.append(
                {
                    "id": name,
                    "metric": metric,
                    "before": left,
                    "after": right,
                    "delta_ms": right - left,
                    "relative_delta": relative,
                }
            )
            if abs(relative) > tolerance:
                failures.append(f"{name}/{metric}: relative drift {relative:.9g} exceeds {tolerance}")
            target = case["metrics"].get(metric)
            if isinstance(target, (int, float)) and math.isfinite(target) and target > 0:
                for label, value in (("before", left), ("after", right)):
                    deviation = (value - target) / target
                    target_deviations.append(
                        {
                            "id": name,
                            "metric": metric,
                            "side": label,
                            "target": target,
                            "actual": value,
                            "relative_deviation": deviation,
                        }
                    )
                    if check_targets and abs(deviation) > tolerance:
                        failures.append(
                            f"{name}/{metric}/{label}: deviation from frozen target {deviation:.9g} exceeds {tolerance}"
                        )
    return {
        "passed": not failures,
        "failures": failures,
        "deltas": deltas,
        "target_deviations": target_deviations,
        **common,
    }


def _percent(value):
    return "n/a" if value is None else f"{value * 100:.6f}"


def _drift_lines(drift, limit=20):
    lines = []
    for label, delta in drift.items():
        for kind in ("missing", "extra", "changed"):
            keys = delta[kind]
            if not keys:
                continue
            lines.append(f"- {label}: {len(keys)} {kind} input(s)")
            lines += [f"  - `{key}`" for key in keys[:limit]]
            if len(keys) > limit:
                lines.append(f"  - ... and {len(keys) - limit} more (full list in comparison.json)")
    return lines


def render_report(result):
    anchor = "frozen targets (--check-targets)" if result["check_targets"] else "none, paired sides only"
    deviations = {
        (row["id"], row["metric"], row["side"]): row["relative_deviation"] for row in result["target_deviations"]
    }
    lines = [
        f"# AFD commit comparison: {'PASS' if result['passed'] else 'FAIL'}",
        "",
        f"Before: `{result['before_commit']}`",
        f"After: `{result['after_commit']}`",
        f"Cases: {result['case_count']}; kind: {result['comparison_kind']}; "
        f"relative tolerance: {result['relative_tolerance']}",
        f"Absolute anchor: {anchor}",
        f"Inputs match manifest baseline: {result['inputs_match_manifest']}",
        "",
        "| Case | Metric | Before ms | After ms | Delta ms | Delta % | Before vs Target % | After vs Target % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["deltas"]:
        key = (row["id"], row["metric"])
        lines.append(
            f"| {row['id']} | {row['metric']} | {row['before']} | {row['after']} | "
            f"{row['delta_ms']:.6f} | {row['relative_delta'] * 100:.6f} | "
            f"{_percent(deviations.get((*key, 'before')))} | {_percent(deviations.get((*key, 'after')))} |"
        )
    drift_lines = _drift_lines(result["input_drift"])
    if drift_lines:
        lines += ["", "## Input fingerprint drift", *drift_lines]
    environment = result["environment_diff"]
    environment_lines = [f"- added `{name}` {version}" for name, version in environment["added"].items()]
    environment_lines += [f"- removed `{name}` {version}" for name, version in environment["removed"].items()]
    environment_lines += [
        f"- changed `{name}` {versions['before']} -> {versions['after']}"
        for name, versions in environment["changed"].items()
    ]
    environment_lines += [
        f"- {key}: `{environment[key]['before']}` -> `{environment[key]['after']}`"
        for key in ("python", "executable")
        if key in environment
    ]
    lines += ["", "## Environment differences (reported, not gated)", *(environment_lines or ["- none"])]
    lines += ["", "## Failures", *([f"- {item}" for item in result["failures"]] or ["- none"])]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--before-python", type=Path)
    parser.add_argument("--after-python", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "tests/e2e/afd_estimate_baseline.json"
    )
    parser.add_argument(
        "--systems-path", type=Path, help="Shared systems directory for identical-input code comparison"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--allow-input-drift", action="store_true", help="Explicit code-and-data comparison, not paired code regression"
    )
    parser.add_argument(
        "--check-targets",
        action="store_true",
        help="Also gate each side against the manifest's frozen targets, not just against each other",
    )
    parser.add_argument("--capture", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.relative_tolerance) or args.relative_tolerance < 0:
        parser.error("relative tolerance must be finite and nonnegative")
    manifest = json.loads(args.manifest.read_text())
    systems = args.systems_path.resolve() if args.systems_path else None
    if args.capture:
        result = capture(args.capture.resolve(), manifest, systems)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 0
    if not args.before or not args.after:
        parser.error("--before and --after are required")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshots = []
    try:
        for label in ("before", "after"):
            repo = getattr(args, label).resolve()
            python = getattr(args, label + "_python") or repo / "python/aisimulate/.venv/bin/python"
            output = args.output / f"{label}.json"
            command = [
                str(python.absolute()),
                str(Path(__file__).resolve()),
                "--capture",
                str(repo),
                "--manifest",
                str(args.manifest.resolve()),
                "--output",
                str(output.resolve()),
            ]
            if systems:
                command += ["--systems-path", str(systems)]
            env = {**os.environ, "PYTHONPATH": str(repo / "python/aisimulate/src"), "PYTHONNOUSERSITE": "1"}
            with (args.output / f"{label}.log").open("w") as log:
                subprocess.run(
                    command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600
                )
            snapshots.append(json.loads(output.read_text()))
        result = compare(*snapshots, manifest, args.relative_tolerance, args.allow_input_drift, args.check_targets)
        result.update(
            before_commit=snapshots[0]["commit"],
            after_commit=snapshots[1]["commit"],
            manifest_sha256=digest(args.manifest),
            harness_sha256=digest(__file__),
            relative_tolerance=args.relative_tolerance,
            case_count=len(manifest["cases"]),
        )
        (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / "report.md").write_text(render_report(result))
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "passed",
                        "case_count",
                        "inputs_equal",
                        "inputs_match_manifest",
                        "comparison_kind",
                        "check_targets",
                        "failures",
                    )
                }
            )
        )
        return 0 if result["passed"] else 2
    except Exception as exc:
        (args.output / "failure.json").write_text(
            json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
