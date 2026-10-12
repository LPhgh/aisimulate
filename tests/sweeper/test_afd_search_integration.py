# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD integration with the generic Sweeper branch, sampler, and replay contract."""

import pytest
from pydantic import ValidationError

from aisimulate.sweeper.afd_parallel import (
    AFDInfeasible,
    AFDParallelConfig,
    AFDReasonCategory,
)
from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.model_hw import ModelHardware, NoViableParallelConfig
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import RunnerCapabilities
from aisimulate.sweeper.sample import unroll_sample
from aisimulate.sweeper.sampler import RandomBranchSampler
from aisimulate.sweeper.search_space import RunnerIncompatibleError, enumerate_branches


def _model_hardware() -> ModelHardware:
    return ModelHardware(
        model_name="example/model",
        hardware_sku="example_sku",
        backend="vllm",
        is_moe=False,
        mla=False,
        enable_wideep=False,
        weight_bytes=1,
        vram_per_gpu=80,
        gpus_per_node=4,
        max_context=4096,
        num_experts=0,
    )


def _topology(*, n_a_nodes: int = 1, n_f_nodes: int = 1) -> dict:
    return {
        "n_a_nodes": n_a_nodes,
        "n_f_nodes": n_f_nodes,
        "tp_a": 2,
        "a_batch_size": 16,
        "num_microbatches": 3,
        "pipeline_model": "full_overlap",
    }


@pytest.mark.parametrize("mode", ["afd"])
def test_afd_rejects_chunked_prefill_control(mode):
    with pytest.raises(ValidationError, match="enable_chunked_prefill is unsupported for AFD"):
        _config(mode, enable_chunked_prefill=True)


def _config(mode: str, **search_overrides) -> SmartSearchConfig:
    search_space = {
        "model_name": "example/model",
        "hardware_sku": "example_sku",
        "deployment_mode": [mode],
        "backend": ["sglang"],
        "gpu_budget": 16,
        "afd_pinned_topologies": [_topology()],
    }
    search_space.update(search_overrides)
    return SmartSearchConfig(
        search_space=search_space,
        workload={"isl": 128, "osl": 32, "concurrency": 8, "num_request_ratio": 2},
    )


def _capabilities(mode: str) -> RunnerCapabilities:
    return RunnerCapabilities(supported_backend_topologies=(("sglang", mode),))


def test_pure_afd_uses_finite_generic_sampler_and_serializes_no_engine(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        lambda *args, **kwargs: _model_hardware(),
    )
    config = _config(
        "afd",
        afd_pinned_topologies=[_topology(), _topology(n_a_nodes=2)],
    )

    (branch,) = enumerate_branches(config, runner_capabilities=_capabilities("afd"))
    suggestion = RandomBranchSampler(branch, seed=7).suggest(1)[0]
    sample = unroll_sample(
        search_space=config.search_space,
        selection=suggestion.selection,
        parallel_config=suggestion.parallel_config,
    )
    deployment = build_backend_deployment(sample, backend_version="test")

    assert branch.flat_parallel_choices
    assert branch.domain_provenance["complete"] is True
    assert len(branch.parallel_configs) == 2
    assert isinstance(suggestion.parallel_config, AFDParallelConfig)
    assert suggestion.projection is not None
    assert sample["used_gpus"] == suggestion.parallel_config.total_gpus
    assert deployment.parallel_config["afd"] == sample["afd"]
    assert deployment.agg_engine_args is None
    assert deployment.prefill_engine_args is None
    assert deployment.decode_engine_args is None
    assert deployment.num_workers == 0
    assert deployment.performance_model_metadata["afd"]["provider"] == "aic-afd"
    assert deployment.performance_model_metadata["afd"]["measurement_required"] is False


def test_afd_plus_pd_is_rejected_loudly_at_config_validation():
    # afd+pd two-pool handoff is retired on the continuous path (future work);
    # the search config must fail loud instead of silently degrading.
    with pytest.raises(ValidationError, match="two-pool prefill/decode handoff"):
        _config("afd+pd", afd_phase="decode")


@pytest.mark.parametrize("pinned", [False, True])
def test_afd_runner_capability_gate_fails_closed(monkeypatch, pinned):
    def unexpected_model_lookup(*args, **kwargs):
        pytest.fail("runner-incompatible AFD must be rejected before model lookup")

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        unexpected_model_lookup,
    )

    with pytest.raises(RunnerIncompatibleError, match=r"runner-incompatible backends=\['sglang'\]"):
        enumerate_branches(
            _config("afd", afd_pinned_topologies=[_topology()] if pinned else [], afd_batch_size_candidates=[16]),
            runner_capabilities=RunnerCapabilities(supported_backend_topologies=(("sglang", "agg"),)),
        )


@pytest.mark.parametrize("pinned", [False, True])
def test_mixed_afd_runner_incompatibility_preserves_explicit_pin_scope(monkeypatch, pinned):
    def unexpected_model_lookup(*args, **kwargs):
        pytest.fail("runner-incompatible AFD must be rejected before model lookup")

    monkeypatch.setattr("aisimulate.sweeper.search_space.resolve_model_hardware", unexpected_model_lookup)
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [
            ReplicaParallelConfig(shape=ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
        ],
    )
    config = _config(
        "afd",
        deployment_mode=["agg", "afd"],
        afd_pinned_topologies=[_topology()] if pinned else [],
        afd_batch_size_candidates=[16],
    )
    if pinned:
        with pytest.raises(NoViableParallelConfig, match="runner-incompatible") as error:
            enumerate_branches(config, runner_capabilities=_capabilities("agg"))
        assert not isinstance(error.value, RunnerIncompatibleError)
    else:
        with pytest.warns(UserWarning, match="runner-incompatible"):
            (branch,) = enumerate_branches(config, runner_capabilities=_capabilities("agg"))
        assert branch.deployment_mode == "agg"


def test_mixed_afd_terminal_failure_defers_warning_and_preserves_runner_details(monkeypatch):
    monkeypatch.setattr("aisimulate.sweeper.search_space.parallel_configs_for", lambda *args, **kwargs: [])
    config = _config("afd", deployment_mode=["afd", "agg"], afd_pinned_topologies=[], afd_batch_size_candidates=[16])

    with pytest.raises(NoViableParallelConfig, match="runner-incompatible") as error:
        enumerate_branches(config, runner_capabilities=_capabilities("agg"))

    assert not isinstance(error.value, RunnerIncompatibleError)


def test_afd_rejects_kv_relative_load_until_capacity_is_exposed(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        lambda *args, **kwargs: _model_hardware(),
    )
    config = SmartSearchConfig(
        search_space=_config("afd").search_space.model_dump(),
        workload={
            "isl": 128,
            "osl": 32,
            "kv_load_ratio": 0.5,
            "num_request_ratio": 2,
        },
    )

    with pytest.raises(AFDInfeasible) as exc_info:
        enumerate_branches(config, runner_capabilities=_capabilities("afd"))

    assert exc_info.value.category is AFDReasonCategory.INVALID_TOPOLOGY
    assert "KV capacity" in exc_info.value.detail


def test_afd_complete_domain_limit_fails_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.resolve_model_hardware",
        lambda *args, **kwargs: _model_hardware(),
    )
    config = _config(
        "afd",
        afd_pinned_topologies=[],
        afd_tp_a_candidates=[1],
        afd_batch_size_candidates=[1],
        afd_microbatch_candidates=[3],
        afd_pipeline_model_candidates=["no_overlap"],
        afd_max_candidates=2,
    )

    with pytest.raises(AFDInfeasible) as exc_info:
        enumerate_branches(config, runner_capabilities=_capabilities("afd"))

    assert exc_info.value.category is AFDReasonCategory.CANDIDATE_LIMIT


def test_afd_combined_product_limit_is_retired_with_afd_plus_pd():
    # The companion product enumeration was afd+pd-only; with the mode
    # retired, the config gate fires before any candidate generation.
    with pytest.raises(ValidationError, match="two-pool prefill/decode handoff"):
        _config("afd+pd", afd_max_candidates=2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"deployment_mode": ["afd"], "backend": ["sglang"]}, "memory-qualified afd_batch_size_candidates"),
        (
            {
                "deployment_mode": ["afd+pd"],
                "afd_phase": "both",
                "afd_batch_size_candidates": [1],
            },
            "two-pool prefill/decode handoff",
        ),
    ],
)
def test_afd_search_contract_rejects_ambiguous_domains(overrides, message):
    with pytest.raises(ValidationError, match=message):
        SmartSearchConfig(
            search_space={
                "model_name": "example/model",
                "hardware_sku": "example_sku",
                **overrides,
            },
            workload={"trace_path": "/tmp/afd-trace.jsonl"},
        )
