# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native per-pool AFD memory-feasibility gate.

The shared backend memory model (``get_partition_memory_usage`` in
``base_backend``) computes activations/KV/NCCL/other per pool with EMPTY
partition ops (zero weights), and the caller adds the Rust-side partition
weight sums (compiled ``Op::weight_bytes`` over the A/F partition,
``sum(weight_bytes) / 1<<30``).

Check composition:

- prefill: ``max_seq_len = max(isl, 1)``, ``num_tokens = 0``, KV multiplier 1;
- decode: ``max_seq_len = isl + osl``, KV multiplier ``num_microbatches``;
- per-pool batch: the A pool checks at
  ``a_micro_batch = ceil(a_batch_size / num_microbatches)``; the F pool serves
  every A worker, so it checks at
  ``b_micro_total = n_a_workers * a_micro_batch`` (``num_tokens`` for decode);
- per-phase weights: ``a_weight_gb`` / ``f_weight_gb`` carry one entry per
  phase (``"prefill"`` = the context partition, ``"decode"`` = the generation
  partition), one per-phase ``sum(op.weight_bytes)``;
- A pool checks with the backend default ``free_gpu_memory_fraction`` (+ the
  backend KV reserved/tolerance params); F pool checks total-only (no
  fraction).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aisimulate.sdk.backends.factory import get_backend
from aisimulate.sdk.config_builders import build_model_config
from aisimulate.sdk.models import (
    get_model,
    resolve_context_fmha_by_data,
    resolve_dsv4_moe_arch,
    resolve_nvfp4_for_system,
)
from aisimulate_core.sdk.memory import kv_cache_budget_bytes
from aisimulate_core.sdk.perf_database import get_database_view, get_latest_database_version


class AFDMemoryInfeasible(RuntimeError):
    """One AFD pool does not fit in GPU memory (the OUT_OF_MEMORY category)."""


def _pool_memory(
    backend,
    database,
    model,
    *,
    partition_weight_gb: float,
    phase: str,
    batch_size: int,
    isl: int,
    osl: int,
    prefix: int,
    max_seq_len: int | None,
    include_kvcache: bool,
    kvcache_multiplier: int,
) -> dict[str, float]:
    if phase == "prefill":
        effective_max_seq_len = max_seq_len if max_seq_len is not None else max(isl, 1)
        num_tokens = 0
        multiplier = 1
    else:
        effective_max_seq_len = max_seq_len if max_seq_len is not None else isl + osl
        num_tokens = batch_size
        multiplier = kvcache_multiplier
    memory = backend.get_partition_memory_usage(
        model,
        database,
        partition_ops=[],
        batch_size=batch_size,
        beam_width=1,
        isl=isl,
        osl=osl,
        num_tokens=num_tokens,
        prefix=prefix,
        max_seq_len=effective_max_seq_len,
        include_kvcache=include_kvcache,
        kvcache_multiplier=multiplier,
    )
    memory = dict(memory)
    memory["weights"] = partition_weight_gb
    memory["total"] = (
        memory["weights"]
        + memory.get("activations", 0.0)
        + memory.get("kvcache", 0.0)
        + memory.get("nccl", 0.0)
        + memory.get("others", 0.0)
    )
    return memory


def _build_pool(
    backend,
    backend_name: str,
    backend_version: str | None,
    a_version: str,
    model_path: str,
    gpus_per_node: int,
    phase: str,
    pool: dict[str, Any],
    *,
    is_a_pool: bool,
):
    """Resolve one pool's database + model. Shared by the fit gate and the
    KV-budget estimator so the two cannot drift."""
    system = pool["system"]
    version = backend_version or (
        a_version if is_a_pool else get_latest_database_version(system=system, backend=backend_name)
    )
    if version is None:
        raise AFDMemoryInfeasible(f"no perf database found for system={system!r}, backend={backend_name!r}")
    database = get_database_view(system, backend_name, version, database_mode="SILICON", allow_missing_data=False)
    if database is None:
        raise AFDMemoryInfeasible(
            f"failed to load the perf database for system={system!r}, backend={backend_name!r}, version={version!r}"
        )
    if is_a_pool:
        tp = pool["tp_a"]
        model_config = build_model_config(tp, 1, 1, tp, 1)
        # The A-worker runs context attention whenever the phase covers
        # prefill ("prefill" or "both") — the F-worker never touches FMHA.
        resolve_context_fmha_by_data(
            model_config, model_path, database, backend_name, is_context_role=phase != "decode"
        )
    else:
        f_tp = pool["n_f_nodes"] * gpus_per_node
        model_config = build_model_config(f_tp, 1, 1, f_tp // pool["f_moe_ep_size"], pool["f_moe_ep_size"])
    resolve_dsv4_moe_arch(model_config, model_path, system_name=system, backend_name=backend_name)
    resolve_nvfp4_for_system(model_config, system, model_path, backend_name=backend_name)
    return database, get_model(model_path, model_config, backend.name.value)


def check_afd_memory_fit(
    *,
    model_path: str,
    backend_name: str,
    backend_version: str | None,
    isl: int,
    osl: int,
    prefix: int,
    # Accepted from the FFI wire but unused: nextn only affects the timing
    # path's verify width, not the memory gate.
    nextn: int,
    a_pool: dict[str, Any],
    f_pool: dict[str, Any],
    gpus_per_node: int,
    a_batch_size: int,
    num_microbatches: int,
    a_weight_gb: Mapping[str, float],
    f_weight_gb: Mapping[str, float],
    phase: str = "both",
    max_seq_len: int | None = None,
) -> None:
    """Raise :class:`AFDMemoryInfeasible` when either pool does not fit.

    ``a_pool`` / ``f_pool`` carry ``system`` + the pool's parallel identity
    (``tp_a``/``n_a_nodes`` for A; ``f_moe_ep_size``/``n_f_nodes`` for F).
    ``a_weight_gb`` / ``f_weight_gb`` map each phase (``"prefill"`` /
    ``"decode"``) to the Rust partition weight sum (GB) of that phase's
    op list.
    """

    backend = get_backend(backend_name)
    # Each pool resolves its own latest version unless the caller pinned one:
    # for heterogeneous pools the A system's latest need not exist for the F
    # pool's system (and vice versa). The A pool's resolution doubles as the
    # version for the A-only free-fraction check below.
    a_version = backend_version or get_latest_database_version(system=a_pool["system"], backend=backend_name)
    if a_version is None:
        raise AFDMemoryInfeasible(f"no perf database found for system={a_pool['system']!r}, backend={backend_name!r}")

    def build_pool(pool: dict[str, Any], *, is_a_pool: bool):
        return _build_pool(
            backend,
            backend_name,
            backend_version,
            a_version,
            model_path,
            gpus_per_node,
            phase,
            pool,
            is_a_pool=is_a_pool,
        )

    num_microbatches = max(int(num_microbatches or 1), 1)
    a_micro_batch_size = max((a_batch_size + num_microbatches - 1) // num_microbatches, 1)
    # The F pool serves every A worker (b_micro_total).
    n_a_workers = max(int(a_pool.get("n_a_nodes", 1)) * int(gpus_per_node) // max(int(a_pool["tp_a"]), 1), 1)
    b_micro_total = n_a_workers * a_micro_batch_size

    check_phases = ("prefill", "decode") if phase == "both" else (phase,)
    free_fraction = backend.get_default_free_gpu_memory_fraction(a_version)

    for check_phase in check_phases:
        for pool, weights, is_a_pool in ((a_pool, a_weight_gb, True), (f_pool, f_weight_gb, False)):
            database, model = build_pool(pool, is_a_pool=is_a_pool)
            memory = _pool_memory(
                backend,
                database,
                model,
                partition_weight_gb=float(weights[check_phase]),
                phase=check_phase,
                batch_size=a_micro_batch_size if is_a_pool else b_micro_total,
                isl=isl,
                osl=osl,
                prefix=prefix,
                max_seq_len=max_seq_len,
                include_kvcache=is_a_pool,
                kvcache_multiplier=num_microbatches,
            )
            mem_capacity_gb = int(database.system_spec["gpu"]["mem_capacity"]) / (1 << 30)
            label = "A" if is_a_pool else "F"
            if memory["total"] >= mem_capacity_gb:
                raise AFDMemoryInfeasible(
                    f"out_of_memory: the model {model_path!r} does not fit in GPU memory on {pool['system']!r} "
                    f"with this AFD topology ({label} pool, phase={check_phase}): "
                    f"total {memory['total']:.2f} GB >= capacity {mem_capacity_gb:.2f} GB; widen tp_a or the "
                    "F pool (which widens the F replica), lower a_batch_size, or use a "
                    "system with more VRAM per GPU"
                )
            if is_a_pool and free_fraction is not None:
                # Shared with the sweep so the two cannot drift: `of_free`
                # (TRT-LLM) scales the FREE pool, while vLLM/SGLang scale TOTAL
                # memory and then subtract the non-KV footprint. The hard-OOM
                # raise above is the early return, so no further precondition
                # belongs here.
                reserved, tolerance = backend.get_kv_cache_memory_check_params()
                kv_gib = memory.get("kvcache", 0.0)
                kv_budget = kv_cache_budget_bytes(
                    capacity=mem_capacity_gb,
                    non_kv=memory["total"] - kv_gib,
                    fraction=free_fraction,
                    of_free=backend.memory_fraction_of_free(),
                    reserved=reserved,
                    tolerance=tolerance,
                )
                if kv_gib > kv_budget:
                    raise AFDMemoryInfeasible(
                        f"out_of_memory: the {label}-pool KV cache ({kv_gib:.2f} GB) exceeds "
                        f"the budget {kv_budget:.2f} GB on {pool['system']!r}; widen tp_a or the F pool, lower "
                        "a_batch_size, or use a system with more VRAM per GPU"
                    )


def estimate_afd_kv_budget_bytes(
    *,
    model_path: str,
    backend_name: str,
    backend_version: str | None,
    isl: int,
    osl: int,
    prefix: int,
    a_pool: dict[str, Any],
    gpus_per_node: int,
    a_batch_size: int,
    num_microbatches: int,
    a_weight_gb: Mapping[str, float],
    phase: str = "both",
    max_seq_len: int | None = None,
) -> tuple[float, float]:
    """Return ``(kv_budget_bytes_per_gpu, kv_bytes_per_token_per_gpu)`` for the A pool.

    Shares the fit gate's memory model so the admission capacity derived from
    this estimate and the OOM check cannot drift. The budget is the minimum
    over the covered phases (the decode phase is retract-relevant: its KV
    footprint grows with generation; a prefill-only pool budgets on prefill).
    The per-token price is measured at unit geometry (one sequence, no
    microbatch multiplier) and is a phase-independent model property.
    """

    backend = get_backend(backend_name)
    a_version = backend_version or get_latest_database_version(system=a_pool["system"], backend=backend_name)
    if a_version is None:
        raise AFDMemoryInfeasible(f"no perf database found for system={a_pool['system']!r}, backend={backend_name!r}")
    database, model = _build_pool(
        backend,
        backend_name,
        backend_version,
        a_version,
        model_path,
        gpus_per_node,
        phase,
        a_pool,
        is_a_pool=True,
    )

    num_microbatches = max(int(num_microbatches or 1), 1)
    a_micro_batch_size = max((a_batch_size + num_microbatches - 1) // num_microbatches, 1)
    check_phases = ("prefill", "decode") if phase == "both" else (phase,)
    free_fraction = backend.get_default_free_gpu_memory_fraction(a_version)
    reserved, tolerance = backend.get_kv_cache_memory_check_params()
    mem_capacity_gb = int(database.system_spec["gpu"]["mem_capacity"]) / (1 << 30)

    budget_gib: float | None = None
    for check_phase in check_phases:
        memory = _pool_memory(
            backend,
            database,
            model,
            partition_weight_gb=float(a_weight_gb[check_phase]),
            phase=check_phase,
            batch_size=a_micro_batch_size,
            isl=isl,
            osl=osl,
            prefix=prefix,
            max_seq_len=max_seq_len,
            include_kvcache=True,
            kvcache_multiplier=num_microbatches,
        )
        kv_gib = memory.get("kvcache", 0.0)
        phase_budget = kv_cache_budget_bytes(
            capacity=mem_capacity_gb,
            non_kv=memory["total"] - kv_gib,
            fraction=free_fraction,
            of_free=backend.memory_fraction_of_free(),
            reserved=reserved,
            tolerance=tolerance,
        )
        budget_gib = phase_budget if budget_gib is None else min(budget_gib, phase_budget)

    effective_seq = max_seq_len if max_seq_len is not None else isl + osl
    unit = _pool_memory(
        backend,
        database,
        model,
        partition_weight_gb=0.0,
        phase="decode",
        batch_size=1,
        isl=isl,
        osl=osl,
        prefix=prefix,
        max_seq_len=max_seq_len,
        include_kvcache=True,
        kvcache_multiplier=1,
    )
    kv_per_token_gib = unit.get("kvcache", 0.0) / max(effective_seq, 1)
    if budget_gib is None or budget_gib <= 0:
        raise AFDMemoryInfeasible(
            f"the A-pool KV budget on {a_pool['system']!r} is not positive ({budget_gib}); "
            "widen tp_a or use a system with more VRAM per GPU"
        )
    if kv_per_token_gib <= 0:
        raise AFDMemoryInfeasible("the A-pool KV per-token estimate is not positive")
    gib = float(1 << 30)
    return budget_gib * gib, kv_per_token_gib * gib
