# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native AFD rank materialization (`_materialize_afd_engine_role`) contracts.

Pins the MTP wiring: the timing config's ``nextn`` only prices the
verify width (cost side); the rank config's ``aic_nextn`` /
``aic_nextn_accept_rates`` build the scheduling core's burst sampler
(benefit side). Both must be emitted together or the decode passes silently
emit one token per pass while paying the verify width. Also pins the
continuous-batching scheduler derivation (max_num_seqs / num_gpu_blocks /
prefill pins) and the fail-loud retirement gates (non-sglang backend,
phase != both).
"""

from __future__ import annotations

import dataclasses
import math

import pytest

import aisimulate.runner as runner
from aisimulate.runner import InvalidRunnerError, _accept_rates_for_expected, _materialize_afd_engine_role
from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec

pytestmark = pytest.mark.unit


def _deployment(metadata: dict, *, phase: str = "both") -> BackendDeploymentSpec:
    topology = {
        "phase": phase,
        "tp_a": 8,
        "n_a_nodes": 1,
        "a_batch_size": 18,
        "f_moe_ep_size": 1,
        "n_f_nodes": 1,
        "gpus_per_node": 8,
        "num_microbatches": 3,
    }
    return BackendDeploymentSpec(
        deployment_mode="afd",
        backend="sglang",
        backend_version="0.5.14",
        parallel_config={"afd": topology},
        performance_model_metadata={"afd": metadata},
    )


def _spec(deployment: BackendDeploymentSpec, workload: dict | None = None) -> ReplaySpec:
    return ReplaySpec(
        backend_deployment=deployment,
        workload=workload if workload is not None else {"isl": 1024, "osl": 128},
        goal={},
    )


def _metadata(**overrides):
    metadata = {
        "provider": "aic-afd",
        "measurement_required": False,
        "model": "test/model",
        "hardware": "h200_sxm",
    }
    metadata.update(overrides)
    return metadata


@pytest.fixture(autouse=True)
def _fake_model_config(monkeypatch):
    monkeypatch.setattr(
        runner,
        "get_model_config_from_model_path",
        lambda model: {"hidden_size": 5120, "layers": 64},
    )


def test_mtp_wires_depth_and_accept_rates_into_the_rank_config():
    spec = _spec(_deployment(_metadata(nextn=2, nextn_accepted=1.5)))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]

    assert engine["aic_nextn"] == 2
    assert engine["aic_nextn_accept_rates"] == _accept_rates_for_expected(2, 1.5, role="afd")
    assert engine["timing_model"]["config"]["nextn"] == 2


def test_non_both_phase_is_rejected_loudly():
    # afd+pd two-pool handoff is retired on the continuous path (future work).
    spec = _spec(_deployment(_metadata(nextn=1, nextn_accepted=0.85), phase="decode"))
    with pytest.raises(InvalidRunnerError, match="two-pool afd\\+pd handoff"):
        _materialize_afd_engine_role(spec.backend_deployment, spec)


def test_non_sglang_backend_is_rejected_loudly():
    deployment = dataclasses.replace(_deployment(_metadata()), backend="trtllm")
    spec = _spec(deployment)
    with pytest.raises(InvalidRunnerError, match="backend='sglang'"):
        _materialize_afd_engine_role(spec.backend_deployment, spec)


def test_mtp_without_nextn_accepted_fails_loudly():
    spec = _spec(_deployment(_metadata(nextn=2)))
    with pytest.raises(InvalidRunnerError, match="nextn_accepted"):
        _materialize_afd_engine_role(spec.backend_deployment, spec)


def test_mtp_depth_above_five_is_rejected():
    spec = _spec(_deployment(_metadata(nextn=6, nextn_accepted=3.0)))
    with pytest.raises(InvalidRunnerError, match="1..=5"):
        _materialize_afd_engine_role(spec.backend_deployment, spec)


def test_no_mtp_leaves_the_rank_config_without_speculation_keys():
    spec = _spec(_deployment(_metadata(nextn=0)))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]

    assert "aic_nextn" not in engine
    assert "aic_nextn_accept_rates" not in engine
    assert engine["timing_model"]["config"]["nextn"] == 0


def test_scheduler_derivation_for_synthetic_workload():
    # max_num_seqs mirrors the pool's in-flight capacity; num_gpu_blocks keeps
    # the radix admission from binding below it; the prefill knobs pin one
    # prompt per pass (the timing identity charges one frozen prefill step).
    spec = _spec(_deployment(_metadata()))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]

    assert engine["max_num_seqs"] == 18
    assert engine["num_gpu_blocks"] == math.ceil(18 * (1024 + 128) * 1.25)
    assert engine["block_size"] == 1
    assert engine["max_model_len"] == 1024 + 128
    assert engine["sglang"] == {"chunked_prefill_size": 1024, "max_prefill_tokens": 1024}
    assert "engine_core" not in engine


def test_scheduler_derivation_uses_the_context_window_when_present():
    spec = _spec(_deployment(_metadata(context_length=8192)))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]

    assert engine["max_num_seqs"] == 18
    assert engine["num_gpu_blocks"] == math.ceil(18 * 8192 * 1.25)
    assert engine["block_size"] == 1
    assert engine["max_model_len"] == 8192
    assert engine["sglang"] == {"chunked_prefill_size": 8192, "max_prefill_tokens": 8192}


def test_context_length_below_isl_plus_osl_is_rejected_loudly():
    # The derivation pins the prompt window AND the KV window at
    # context_length, so a window that cannot hold the declared workload would
    # silently chunk the prompt and undersize num_gpu_blocks.
    spec = _spec(_deployment(_metadata(context_length=512)))
    with pytest.raises(InvalidRunnerError, match="cannot hold isl\\+osl"):
        _materialize_afd_engine_role(spec.backend_deployment, spec)


def test_variable_length_traffic_is_exempt_from_the_window_lower_bound():
    # Sampled lengths are upper bounds, not per-request lengths, so the
    # window check must not fire; SglangCore's max_model_len gate handles a
    # genuinely over-long single prompt instead.
    spec = _spec(
        _deployment(_metadata(context_length=512)),
        workload={"isl": 1024, "osl": 128, "random_range_ratio": 0.5},
    )
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]

    assert engine["max_model_len"] == 512
    assert engine["num_gpu_blocks"] == math.ceil(18 * 512 * 1.25)


def test_afd_capability_surface_is_exactly_sglang_pure_afd():
    # Guards against the advertised capability set drifting away from what
    # `_materialize_engine_execution_spec` actually accepts: every other
    # backend/topology AFD combination is fail-loud retired.
    topologies = runner.EngineReplayRunnerFactory().capabilities().supported_backend_topologies
    assert {entry for entry in topologies if entry[1].startswith("afd")} == {("sglang", "afd")}


def test_trace_workload_without_context_length_reserves_the_model_window(monkeypatch):
    # The memory gate must not size KV from the placeholder isl/osl (1/2) of
    # variable-length traffic: reserve the model's full context, matching the
    # compiler path's "max" resolution.
    monkeypatch.setattr(runner, "resolve_model_context_length", lambda model: 40960)
    spec = _spec(_deployment(_metadata()), workload={"trace_path": "/tmp/trace.jsonl"})
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]

    assert engine["timing_model"]["config"]["max_seq_len"] == 40960
    assert engine["num_gpu_blocks"] == math.ceil(18 * 40960 * 1.25)
    assert engine["block_size"] == 1
    assert engine["max_model_len"] == 40960


def test_explicit_context_length_wins_for_trace_workloads(monkeypatch):
    monkeypatch.setattr(runner, "resolve_model_context_length", lambda model: 40960)
    spec = _spec(_deployment(_metadata(context_length=8192)), workload={"trace_path": "/tmp/trace.jsonl"})
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]

    assert engine["timing_model"]["config"]["max_seq_len"] == 8192


def test_synthetic_workload_keeps_the_isl_osl_window(monkeypatch):
    def no_resolve(model):
        raise AssertionError("resolve_model_context_length must not be called for synthetic traffic")

    monkeypatch.setattr(runner, "resolve_model_context_length", no_resolve)
    spec = _spec(_deployment(_metadata()))
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]

    assert engine["timing_model"]["config"]["max_seq_len"] is None


def test_fixed_synthetic_traffic_keeps_the_frozen_decode_identity():
    spec = _spec(_deployment(_metadata()), workload={"isl": 1024, "osl": 128, "random_range_ratio": 1.0})
    config = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]["timing_model"]["config"]

    assert config["frozen_decode_identity"] is True


def test_variable_length_synthetic_traffic_drops_the_frozen_decode_identity(monkeypatch):
    # random_range_ratio < 1.0 samples per-request lengths around the declared
    # isl/osl, so pricing every decode at the frozen identity would silently
    # reuse a measurement taken at a different sequence length. The declared
    # isl/osl are still real upper bounds, so the KV window must NOT fall back
    # to the whole model context the way placeholder-length traffic does.
    def no_resolve(model):
        raise AssertionError("declared isl/osl are real bounds; the model window must not be reserved")

    monkeypatch.setattr(runner, "resolve_model_context_length", no_resolve)
    spec = _spec(_deployment(_metadata()), workload={"isl": 1024, "osl": 128, "random_range_ratio": 0.5})
    engine = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]

    assert engine["timing_model"]["config"]["frozen_decode_identity"] is False
    assert engine["timing_model"]["config"]["max_seq_len"] is None
    assert engine["num_gpu_blocks"] == math.ceil(18 * (1024 + 128) * 1.25)
    assert engine["block_size"] == 1
    assert engine["max_model_len"] == 1024 + 128


def test_placeholder_length_traffic_drops_the_frozen_decode_identity(monkeypatch):
    monkeypatch.setattr(runner, "resolve_model_context_length", lambda model: 40960)
    spec = _spec(_deployment(_metadata()), workload={"trace_path": "/tmp/trace.jsonl"})
    config = _materialize_afd_engine_role(spec.backend_deployment, spec)["rank"]["timing_model"]["config"]

    assert config["frozen_decode_identity"] is False
