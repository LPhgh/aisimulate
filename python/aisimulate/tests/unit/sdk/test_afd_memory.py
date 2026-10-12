# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract of the native AFD memory-feasibility gate.

Pins the two per-pool batch compositions:

- the F pool checks at ``b_micro_total = n_a_workers * a_micro_batch`` (the F
  workers serve every A worker), while the A pool checks at ``a_micro_batch``;
- each phase's check consumes that phase's own partition weight sum
  (prefill -> context partition, decode -> generation partition).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import aisimulate.sdk.afd_memory as afd_memory
from aisimulate_core.sdk.memory import kv_cache_budget_bytes

pytestmark = pytest.mark.unit

_CAPACITY_BYTES = 140 * (1 << 30)
_CAPACITY_GB = _CAPACITY_BYTES / (1 << 30)


class _FakeBackend:
    """Captures the resolved per-call memory-usage kwargs (num_tokens etc.)."""

    name = SimpleNamespace(value="trtllm")

    def __init__(
        self,
        *,
        free_fraction: float | None = None,
        of_free: bool = False,
        kv_check_params: tuple[float, float] = (0.0, 0.0),
        usage: dict[str, float] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self._free_fraction = free_fraction
        self._of_free = of_free
        self._kv_check_params = kv_check_params
        self._usage = usage if usage is not None else {"activations": 0.1, "kvcache": 0.1, "nccl": 0.0, "others": 0.0}

    def get_default_free_gpu_memory_fraction(self, version):
        return self._free_fraction

    def get_kv_cache_memory_check_params(self):
        return self._kv_check_params

    def memory_fraction_of_free(self):
        return self._of_free

    def get_partition_memory_usage(self, model, database, **kwargs):
        self.calls.append(kwargs)
        return dict(self._usage)


def _install(monkeypatch, **backend_kwargs) -> tuple[_FakeBackend, list[dict]]:
    backend = _FakeBackend(**backend_kwargs)
    monkeypatch.setattr(afd_memory, "get_backend", lambda name: backend)
    monkeypatch.setattr(
        afd_memory,
        "get_database_view",
        lambda *a, **k: SimpleNamespace(system_spec={"gpu": {"mem_capacity": _CAPACITY_BYTES}}),
    )
    monkeypatch.setattr(afd_memory, "build_model_config", lambda *a, **k: object())
    monkeypatch.setattr(afd_memory, "resolve_context_fmha_by_data", lambda *a, **k: None)
    monkeypatch.setattr(afd_memory, "resolve_dsv4_moe_arch", lambda *a, **k: None)
    monkeypatch.setattr(afd_memory, "resolve_nvfp4_for_system", lambda *a, **k: None)
    monkeypatch.setattr(afd_memory, "get_model", lambda *a, **k: object())

    # Spy on _pool_memory to see the phase/batch/weight selection of the
    # check loop (the real body still runs against the fake backend).
    captured: list[dict] = []
    real_pool_memory = afd_memory._pool_memory

    def spy_pool_memory(backend_arg, database, model, **kwargs):
        captured.append(dict(kwargs))
        return real_pool_memory(backend_arg, database, model, **kwargs)

    monkeypatch.setattr(afd_memory, "_pool_memory", spy_pool_memory)
    return backend, captured


def test_f_pool_batch_scales_with_n_a_workers_and_weights_follow_phase(monkeypatch):
    backend, captured = _install(monkeypatch)

    # a_micro_batch = ceil(12 / 3) = 4; n_a_workers = 2 nodes * 8 GPUs / tp_a=4
    # = 4; b_micro_total = 4 * 4 = 16.
    afd_memory.check_afd_memory_fit(
        model_path="Qwen/Qwen3-32B",
        backend_name="trtllm",
        backend_version="1.3.0",
        isl=1024,
        osl=128,
        prefix=0,
        nextn=0,
        a_pool={"system": "h200_sxm", "tp_a": 4, "n_a_nodes": 2},
        f_pool={"system": "h200_sxm", "f_moe_ep_size": 1, "n_f_nodes": 1},
        gpus_per_node=8,
        a_batch_size=12,
        num_microbatches=3,
        a_weight_gb={"prefill": 1.0, "decode": 2.0},
        f_weight_gb={"prefill": 3.0, "decode": 4.0},
        phase="both",
        max_seq_len=None,
    )

    assert [(entry["phase"], entry["batch_size"], entry["partition_weight_gb"]) for entry in captured] == [
        ("prefill", 4, 1.0),  # A pool: micro batch, context-partition weights
        ("prefill", 16, 3.0),  # F pool: b_micro_total, context-partition weights
        ("decode", 4, 2.0),  # A pool: micro batch, generation-partition weights
        ("decode", 16, 4.0),  # F pool: b_micro_total, generation-partition weights
    ]
    # A-pool-only KV; the phase -> multiplier mapping happens inside
    # _pool_memory (prefill charges 1, decode num_microbatches).
    assert [entry["include_kvcache"] for entry in captured] == [True, False, True, False]
    # The phase composition inside _pool_memory: prefill charges no
    # tokens at max(isl, 1) with multiplier 1; decode charges each pool's own
    # batch at isl+osl with the microbatch multiplier.
    assert [(call["num_tokens"], call["max_seq_len"], call["kvcache_multiplier"]) for call in backend.calls] == [
        (0, 1024, 1),
        (0, 1024, 1),
        (4, 1152, 3),
        (16, 1152, 3),
    ]


def test_single_phase_role_checks_only_that_phase(monkeypatch):
    _backend, captured = _install(monkeypatch)

    afd_memory.check_afd_memory_fit(
        model_path="Qwen/Qwen3-32B",
        backend_name="trtllm",
        backend_version="1.3.0",
        isl=1024,
        osl=128,
        prefix=0,
        nextn=0,
        a_pool={"system": "h200_sxm", "tp_a": 8, "n_a_nodes": 1},
        f_pool={"system": "h200_sxm", "f_moe_ep_size": 1, "n_f_nodes": 1},
        gpus_per_node=8,
        a_batch_size=18,
        num_microbatches=3,
        a_weight_gb={"prefill": 1.0, "decode": 2.0},
        f_weight_gb={"prefill": 3.0, "decode": 4.0},
        phase="decode",
        max_seq_len=None,
    )

    # n_a_workers = 1 * 8 / 8 = 1 -> b_micro_total == a_micro_batch == 6.
    assert [(entry["phase"], entry["batch_size"], entry["partition_weight_gb"]) for entry in captured] == [
        ("decode", 6, 2.0),
        ("decode", 6, 4.0),
    ]


def _check_hetero() -> None:
    afd_memory.check_afd_memory_fit(
        model_path="Qwen/Qwen3-32B",
        backend_name="trtllm",
        backend_version=None,
        isl=1024,
        osl=128,
        prefix=0,
        nextn=0,
        a_pool={"system": "h200_sxm", "tp_a": 8, "n_a_nodes": 1},
        f_pool={"system": "b200_sxm", "f_moe_ep_size": 1, "n_f_nodes": 1},
        gpus_per_node=8,
        a_batch_size=18,
        num_microbatches=3,
        a_weight_gb={"prefill": 1.0, "decode": 2.0},
        f_weight_gb={"prefill": 3.0, "decode": 4.0},
        phase="prefill",
        max_seq_len=None,
    )


def test_hetero_pools_resolve_their_own_latest_versions(monkeypatch):
    backend, _captured = _install(monkeypatch)
    monkeypatch.setattr(
        afd_memory,
        "get_latest_database_version",
        lambda *, system, backend: {"h200_sxm": "1.3.0", "b200_sxm": "0.24.0"}[system],
    )
    views: list[tuple[str, str]] = []

    def record_view(system, backend_name, version, **kwargs):
        views.append((system, version))
        return SimpleNamespace(system_spec={"gpu": {"mem_capacity": _CAPACITY_BYTES}})

    monkeypatch.setattr(afd_memory, "get_database_view", record_view)
    fraction_versions: list[str] = []
    monkeypatch.setattr(
        backend,
        "get_default_free_gpu_memory_fraction",
        lambda version: fraction_versions.append(version),
    )

    _check_hetero()

    # Each pool's database view resolves its own system's latest version; the
    # A-only free-fraction check keys off the A pool's version.
    assert views == [("h200_sxm", "1.3.0"), ("b200_sxm", "0.24.0")]
    assert fraction_versions == ["1.3.0"]


def test_explicit_backend_version_applies_to_both_pools(monkeypatch):
    _backend, _captured = _install(monkeypatch)

    def no_latest(*args, **kwargs):
        raise AssertionError("get_latest_database_version must not be called")

    monkeypatch.setattr(afd_memory, "get_latest_database_version", no_latest)
    views: list[tuple[str, str]] = []

    def record_view(system, backend_name, version, **kwargs):
        views.append((system, version))
        return SimpleNamespace(system_spec={"gpu": {"mem_capacity": _CAPACITY_BYTES}})

    monkeypatch.setattr(afd_memory, "get_database_view", record_view)

    afd_memory.check_afd_memory_fit(
        model_path="Qwen/Qwen3-32B",
        backend_name="trtllm",
        backend_version="9.9.9",
        isl=1024,
        osl=128,
        prefix=0,
        nextn=0,
        a_pool={"system": "h200_sxm", "tp_a": 8, "n_a_nodes": 1},
        f_pool={"system": "b200_sxm", "f_moe_ep_size": 1, "n_f_nodes": 1},
        gpus_per_node=8,
        a_batch_size=18,
        num_microbatches=3,
        a_weight_gb={"prefill": 1.0, "decode": 2.0},
        f_weight_gb={"prefill": 3.0, "decode": 4.0},
        phase="prefill",
        max_seq_len=None,
    )

    assert views == [("h200_sxm", "9.9.9"), ("b200_sxm", "9.9.9")]


def _check_a_pool(*, a_weight_gb: float, phase: str = "prefill") -> None:
    """Run the gate with a controllable A-pool weight sum (F pool stays tiny)."""

    afd_memory.check_afd_memory_fit(
        model_path="Qwen/Qwen3-32B",
        backend_name="sglang",
        backend_version="0.5.14",
        isl=1024,
        osl=128,
        prefix=0,
        nextn=0,
        a_pool={"system": "h200_sxm", "tp_a": 8, "n_a_nodes": 1},
        f_pool={"system": "h200_sxm", "f_moe_ep_size": 1, "n_f_nodes": 1},
        gpus_per_node=8,
        a_batch_size=18,
        num_microbatches=3,
        a_weight_gb={"prefill": a_weight_gb, "decode": a_weight_gb},
        f_weight_gb={"prefill": 1.0, "decode": 1.0},
        phase=phase,
        max_seq_len=None,
    )


def _install_kv(monkeypatch, *, kvcache: float, of_free: bool, reserved: float = 0.0, tolerance: float = 0.0) -> None:
    _install(
        monkeypatch,
        free_fraction=0.9,
        of_free=of_free,
        kv_check_params=(reserved, tolerance),
        usage={"activations": 0.0, "kvcache": kvcache, "nccl": 0.0, "others": 0.0},
    )


def test_of_total_kv_budget_subtracts_non_kv(monkeypatch):
    """sglang/vLLM cap TOTAL memory, so the budget must subtract the non-KV footprint.

    A pool: weights 100 GiB + KV 30 GiB = 130 GiB total on a 140 GiB device at
    fraction 0.9. The shared formula gives ``140 * 0.9 - 100 = 26`` GiB, so the
    30 GiB KV cache is over budget; a formula that forgets ``- non_kv`` compares
    against 126 GiB and waves it through.
    """
    _install_kv(monkeypatch, kvcache=30.0, of_free=False)

    expected = kv_cache_budget_bytes(capacity=_CAPACITY_GB, non_kv=100.0, fraction=0.9, of_free=False)
    assert expected == pytest.approx(26.0)

    with pytest.raises(afd_memory.AFDMemoryInfeasible, match="KV cache"):
        _check_a_pool(a_weight_gb=100.0)


def test_kv_check_runs_below_the_fraction_of_total(monkeypatch):
    """The KV gate must not be gated on ``total >= capacity * fraction``.

    With tolerance 0.1 the budget is ``140 * 0.9 * 0.9 - 100 = 13.4`` GiB, so a
    20 GiB KV cache is over budget even though the 120 GiB total is still under
    ``140 * 0.9 = 126`` GiB.
    """
    _install_kv(monkeypatch, kvcache=20.0, of_free=False, tolerance=0.1)

    with pytest.raises(afd_memory.AFDMemoryInfeasible, match="KV cache"):
        _check_a_pool(a_weight_gb=100.0)


def test_of_free_budget_and_strict_boundary(monkeypatch):
    """TRT-LLM applies the fraction to the FREE pool and rejects only above it."""
    _install_kv(monkeypatch, kvcache=45.0, of_free=True)

    # non_kv = 135 - 45 = 90; budget = (140 - 90) * 0.9 = 45 -> 45 is not > 45.
    assert kv_cache_budget_bytes(capacity=_CAPACITY_GB, non_kv=90.0, fraction=0.9, of_free=True) == pytest.approx(45.0)
    _check_a_pool(a_weight_gb=90.0)


def test_of_free_budget_rejects_one_gib_over(monkeypatch):
    _install_kv(monkeypatch, kvcache=46.0, of_free=True)

    with pytest.raises(afd_memory.AFDMemoryInfeasible, match="KV cache"):
        _check_a_pool(a_weight_gb=90.0)


@pytest.mark.parametrize(
    ("of_free", "reserved", "tolerance", "weights", "kvcache"),
    [
        (False, 0.0, 0.0, 100.0, 25.0),
        (False, 0.0, 0.0, 100.0, 30.0),
        (False, 0.05, 0.05, 90.0, 30.0),
        (False, 0.0, 0.1, 100.0, 20.0),
        (True, 0.0, 0.0, 90.0, 44.0),
        (True, 0.0, 0.0, 90.0, 45.0),
        (True, 0.0, 0.0, 90.0, 46.0),
        (True, 0.1, 0.0, 60.0, 40.0),
    ],
)
def test_kv_gate_agrees_with_the_shared_budget_formula(monkeypatch, of_free, reserved, tolerance, weights, kvcache):
    """Anti-divergence guard: the gate must decide exactly what the shared helper says.

    ``kv_cache_budget_bytes`` exists so the estimate and the sweep share one
    formula; this pins the AFD gate to it across both fraction semantics.
    """
    _install_kv(monkeypatch, kvcache=kvcache, of_free=of_free, reserved=reserved, tolerance=tolerance)

    total = weights + kvcache
    if total >= _CAPACITY_GB:
        pytest.skip("hard-OOM path raises before the KV budget check")
    budget = kv_cache_budget_bytes(
        capacity=_CAPACITY_GB,
        non_kv=total - kvcache,
        fraction=0.9,
        of_free=of_free,
        reserved=reserved,
        tolerance=tolerance,
    )
    if kvcache > budget:
        with pytest.raises(afd_memory.AFDMemoryInfeasible, match="KV cache"):
            _check_a_pool(a_weight_gb=weights)
    else:
        _check_a_pool(a_weight_gb=weights)
