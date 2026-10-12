---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AFD Topology Contract
subtitle: Attention-FFN parallel shapes and complete enumeration
---

> [!WARNING]
> **Experimental — analytical only, not deployable.** AFD prediction emits a
> replay contract and a qualification artifact, but no physical serving adapter
> or native launch renderer exists yet, so an AFD prediction cannot be deployed.
> Coverage is partial: `backend: sglang` only (pure `phase: both` or the
> two-pool `afd+pd` form with one monolithic companion). The qualification
> artifact pins `native_deployment_supported: false` and `launch.supported: false`
> so it is never mistaken for a runnable backend manifest.

> [!NOTE]
> **Native implementation.** AFD replay runs through the standard
> `run_replay_json` path with the `"aic-afd"` timing provider
> (`AISIMULATE_AFD_NATIVE=1` is the default); the retired Python island is
> removed, so the native path is the only path. The AFD pool runs on the
> monolithic SGLang scheduling core (continuous batching: mid-flight
> admission + retraction) with the timing provider injected; `backend: sglang`
> is required (scheduling and perf data share the sglang family). The two-pool
> `afd+pd` handoff runs on the same core: the AFD pool serves one phase as a
> disagg role paired with a monolithic companion on the generic handoff
> surface. See the ADRs below for the migration history and the retired
> `AfdCore` / `engine_core` axis.

Attention-FFN Disaggregation (AFD) places attention operations on an A-worker pool and FFN/MoE
operations on an F-worker pool. `aisimulate.sweeper.afd_parallel` provides the backend-neutral
parallel shape, validation, GPU accounting, and complete finite enumeration. Performance
measurement, staged evaluation, generic search integration, replay, and deployment generation
belong to later layers.

## Legacy Mapping

| Legacy AIC field or result | Sweeper AFD contract |
|---|---|
| `n_a_nodes` / `n_f_nodes` | `AFDTopology.n_a_nodes` / `n_f_nodes` |
| `tp_a` | `AFDTopology.tp_a` |
| derived `tp_f` | `AFDTopology.ffn_tp` (`n_f_nodes * gpus_per_node`) |
| `f_moe_ep_size` | `AFDTopology.f_moe_ep_size` |
| `a_batch_size` | `AFDTopology.a_batch_size` |
| `num_microbatches` | `AFDTopology.num_microbatches` |
| `pipeline_model` | `no_overlap`, `partial_overlap`, or `full_overlap` (legacy aliases: `serial`, `conservative`, `optimistic`) |
| `combined_with_pd` | whether a later layer must add the complementary P/D phase |
| `num_total_gpus` | A GPUs + F GPUs for this AFD topology |
| AFD rejection logs | `AFDInfeasible.category`, detail, and provenance |

F-side semantics are intentionally strict: one F replica spans all F GPUs, so F TP equals
the number of F workers. A-side workers are the A GPU count divided by A TP. Expert parallelism
must divide both F TP and the model expert count when the latter is known.

## Pinned and Searched Shapes

Complete `AFDTopology` objects passed in `AFDSearchConfig.pinned_topologies` are preserved exactly.
When that tuple is empty, `enumerate_afd_topologies` searches:

- A-node and F-node counts within the GPU budget;
- A-worker TP and batch size;
- F-worker MoE EP, including `n_f_nodes` and `ffn_tp` symbolic choices;
- microbatch count and pipeline model; and
- A:F node ratio.

`total_gpus` is an upper-bound budget, not an exact allocation. Because AFD enumeration is
node-granular, a remainder smaller than `gpus_per_node` may remain unused while the enumerator
still covers every full-node topology within the budget.

The enumerator preserves the legacy canonical order and evaluates the complete finite domain.
If the domain exceeds `max_candidates`, it fails with `candidate_limit` instead of returning a
partial result.

```python
from aisimulate.sweeper import AFDSearchConfig, enumerate_afd_topologies

domain = enumerate_afd_topologies(
    AFDSearchConfig(
        total_gpus=64,
        gpus_per_node=8,
        is_moe=True,
        num_experts=256,
        tp_a_candidates=(2, 4, 8),
        a_batch_size_candidates=(64, 128),
        f_moe_ep_size_candidates=("n_f_nodes", "ffn_tp"),
        microbatch_candidates=(3, 4),
    )
)
```

## Performance Model

AFD has no compile-time measurement step. The `"aic-afd"` timing provider
(`AfdForwardPassPerfModel`, `crates/core/src/perfmodel/afd/model.rs`) is built when the replay role is
materialized and evaluates the A/F pools on demand, so a candidate carries no frozen layer times.

The provider holds one perf-database view per pool system (`a_pool.system`, `f_pool.system`), each
constructed through `ForwardPassPerfModel::best_available` with the caller's `estimator_config`
passed through unchanged. A heterogeneous pair therefore reads each pool's own tables; when the two
systems differ, the F pool's inherited gemm/moe quant modes are cleared and re-resolved against
`f_system` so a checkpoint is never priced with the other pool's quantization.

The A↔F traffic is modeled by four native `Op` variants (`AfdTransfer`, `AfdFAllGather`,
`AfdFReduceScatter`, `AfdCombine`, appended to the `Op` enum at engine-spec schema 22). Op
partitioning between the pools lives in `crates/core/src/perfmodel/afd/partition.rs`.

Two contracts are enforced at build time:

- `comm_overhead_factor` is applied **exactly once**, at pipeline evaluation; the underlying
  measurements stay unscaled.
- The per-pool memory gate (`aisimulate/sdk/afd_memory.py`, invoked from the provider constructor)
  must pass for every configured phase, sharing its KV budget formula with the sweep
  (`kv_cache_budget_bytes`) so the two cannot drift. An infeasible pool fails the candidate loudly
  with the `out_of_memory` category.

## Staged Evaluation

One pass of the A→A2F→F→F2A layer ping-pong collapses **closed-form into a single `step_latency`**;
the replay event loop sees one pass, not per-layer events. `evaluate_afd_phase`
(`crates/core/src/engine/scheduler/afd/pipeline.rs`) applies the pipeline regimes:

- full_overlap (legacy `optimistic`): `max(A, F, A_to_F + F_to_A)`, with the minimum-microbatch check;
- partial_overlap (legacy `conservative`): `max(A + A_to_F, F + F_to_A)`; and
- no_overlap (legacy `serial`): the sum of all compute and transfer stages.

Global step latency includes pipeline fill and every microbatch-layer cadence.

The pool itself runs on the **monolithic SGLang continuous-batching core** with the provider injected
through the standard `PerfModel::External` seam, so admission happens mid-flight and decode can be
retracted; there is no AFD-specific scheduler loop. The scheduler surface is derived from the
topology: `max_num_seqs = n_a_workers * a_batch_size`, `num_gpu_blocks` sized so `max_num_seqs` is
the only binding admission limit, and `chunked_prefill_size` / `max_prefill_tokens` pinned at the
prompt window so a prompt pre-fills in one pass.

Decode pricing depends on whether per-request lengths vary:

- **fixed synthetic traffic** (`random_range_ratio: 1.0`) keeps the island's frozen stride-averaged
  decode identity — every pass pays the full-batch step measured once at provider construction;
- **trace / session / agentic traffic and synthetic traffic with `random_range_ratio < 1.0`** use the
  instantaneous per-pass evaluation at the current mean KV position, with the effective microbatch
  count clamped to what the pool-wide in-flight batch can fill (`m' = clamp(ceil(per_worker /
  micro_full), 1, num_microbatches)`; `m' == 1` degrades to the serial cadence).

Prefill always evaluates at the actual batch geometry (`mean_isl` / `mean_prefix`), so prefix-cached
workloads are priced at their real suffix.

A topology covering both phases executes prefill and decode through the same A/F pool. With
`combined_with_pd: true` and a single `phase` (`prefill` or `decode`), the deployment becomes
`afd+pd`: the AFD pool serves exactly one phase as a disaggregated role and a monolithic
companion serves the opposite phase on the generic two-pool handoff surface
(`ReplayTopology::Disaggregated` + `ReserveDestination` / `ActivateDestination` / source holds).
Per-worker GPU accounting reports the A/F pool-pair total on the AFD role and the monolithic
`dp * tp` product on the companion.

## Generic Sweeper Domain

The internal `SmartSearchConfig` schema accepts `deployment_mode: [afd]` for a pure A/F pool, using
one finite `parallel_config_choice` dimension in the standard sampler. Two further constraints come
from the continuous path: `backend` must be exactly `["sglang"]` (scheduling flavor and the perf-DB
family the provider queries move together), and pure `afd` requires `afd_phase: both`.
`deployment_mode: [afd+pd]` requires `afd_phase: prefill|decode` and searches the two-pool product —
each A/F topology times the opposite-phase companion shapes pinned in
`afd_companion_parallel_configs` (flat `{tp, attention_dp?, moe_tp?, moe_ep?, pp?, replicas?}`
dicts; empty means the KV-feasible companion domain is enumerated). The unified recommend input
expresses the same pin through `engine.workers.<companion>.parallelism.preset`.

Use `afd_pinned_topologies` to provide concrete A/F shapes, or provide an explicit,
memory-qualified `afd_batch_size_candidates` list with the other `afd_*_candidates` fields to
generate the domain. `afd_max_candidates` rejects an oversized domain instead of silently
truncating it. Pure AFD has no ordinary engine argument payload.

KV-relative traffic load is intentionally rejected for AFD in this layer because the A/F pools do
not yet expose scheduler-visible KV capacity. Use a synthetic request rate or absolute concurrency
with concrete `isl` and `osl`.
See [Sweeper Configuration](configuration.md#attention-ffn-disaggregation) for an internal example.

## Analytical Replay

AFD replay runs through the standard `run_replay_json` path: `EngineReplayRunner` materializes the
AFD role with the `"aic-afd"` timing provider and the ordinary replay runtime drives admission, KV,
telemetry, and reporting. There is no measurement contract to consume and no AFD-specific replay
loop. Pure `afd` replay requires `phase: both`.

This preserves arrival and queueing delay in TTFT and end-to-end latency. TPOT is the interval
from the first output token to completion, including decode queueing, divided by `OSL - 1`;
it is zero for a single output token. Reports include throughput, goodput, GPU-hours, per-request
latency on request, batch/pass counts, and exact A/F GPU accounting.

Trace and prefix-cached workloads are supported. What varies is the decode pricing identity, not
admissibility: fixed synthetic traffic (`random_range_ratio: 1.0`) is priced at the frozen
stride-averaged step, while trace/session/agentic traffic and synthetic traffic with
`random_range_ratio < 1.0` are priced per pass at the actual mean KV position (see
[Staged Evaluation](#staged-evaluation)). Variable-length traffic without an explicit
`context_length` reserves the model's full context window for the memory gate, so it cannot pass a
deployment that would OOM in reality.

## Qualification Artifacts

Every successful public AFD prediction writes `afd-replay-spec.json` and
`afd-qualification.json`. The replay artifact freezes the complete runner input; the qualification
artifact checks the native provider identity, topology/provenance agreement, and A/F GPU
accounting. Its logical plan describes the attention and FFN pools and
routing order, while `native_deployment_supported: false` and `launch.supported: false` prevent it
from being mistaken for a runnable backend manifest.

The artifacts have deterministic JSON ordering and a SHA-256 link from the qualification record to
the replay spec. They qualify the analytical path for regression and release testing. They do not
remove the need for `aiconfigurator cli generate` where native Dynamo, llm-d, or FPM output is
required.

## Infeasibility and Provenance

Every topology failure uses a stable category such as `invalid_topology`, `gpu_budget`,
`expert_divisibility`, `out_of_memory`, or `candidate_limit`. Searched enumeration reports filter counts, the
canonical candidate dimensions, and whether its finite domain was complete. Pinned enumeration
identifies `AFDSearchConfig.pinned_topologies` as its source and does not claim generated candidate
dimensions. Each topology records its phase, parallel shape, and lossless A/F GPU accounting;
generator provenance is recorded by the enumeration that created it, not by manually constructed
topology objects.

## Terminal State ADR: AFD as an Execution-Model Axis

> Status: implemented, then **partially superseded by [ADR v2](#terminal-state-adr-v2-afd-on-the-continuous-sglang-core)
> below** — the `engine_core` axis and the batch-locked `AfdCore` described here
> are deleted; the perfmodel layer, the comm-op schema plan, and the
> conservation invariants still hold. Kept as the record of the migration from
> the Python island (`runner.py::_run_afd_replay` + `AFDForegroundEngine` +
> compile-time-frozen `measure_afd_layer_times`) to a first-class Rust surface.

**Decision.** AFD is not a fourth `backend`. It is an execution-model axis
orthogonal to both topology and perf-backend:
`topology(agg|disagg)` x `engine_core(monolithic|afd)` x
`perf_backend(vllm|sglang|trtllm)`. Pure AFD is
`ReplayTopology::Aggregated` + `engine_core=afd`; `afd+pd` is
`ReplayTopology::Disaggregated` with one role at `engine_core=afd`. No new
`ReplayTopology` variant is introduced. The serialized field is named
`engine_core` — never `execution_model`, which is already taken at the
traffic level for the agentic target-model name (`python.rs`
`RuntimeTraffic.execution_model`).

**Model X+ scheduling.** The A->A2F->F->F2A layer ping-pong collapses
closed-form into a single `step_latency` inside one pass (no per-layer
replay events), implemented by a new `EngineCore::Afd(AfdCore)` on the
existing `RankEngine` contract. The event loop reuses `AggRuntimeImpl` /
`DisaggRuntimeImpl` (heap, admission, KV, DP barrier, deadlock detection,
telemetry). Pool heterogeneity lives in the timing layer: the `"aic-afd"`
provider (`AfdForwardPassPerfModel`) holds one perf-DB view per pool system
(`a_pool.system`, `f_system`), each constructed through
`ForwardPassPerfModel::best_available` — the single production interface —
with `estimator_config` passed through unchanged.

**Conservation invariants**:

1. `comm_overhead_factor` is applied exactly once, at pipeline evaluation;
   measurements stay unscaled (`comm_overhead_factor=1.0`).
2. `gpus_per_node` agrees three ways per pool: `resolve_model_hardware`,
   the pool spec, and the pool's perf DB. Homogeneous
   (`a_system == f_system`) must degrade bit-exactly to today's
   single-system numbers.
3. `phase=both` forbids pairing; a measurement covers the AFD pool alone
   (`combined_with_pd=False`).
4. The no_overlap/partial_overlap/full_overlap (legacy
   serial/conservative/optimistic) cycle/fill/step formulas are carried
   over verbatim.

**Schema plan.** The four AFD comm ops (`AfdTransferOp`,
`AfdFAllGatherOp`, `AfdFReduceScatterOp`, `AfdCombineOp`) are APPENDED to
the `Op` enum (bincode indices are positional; never insert mid-enum) and
`ENGINE_SPEC_SCHEMA_VERSION` bumps 21 -> 22. The Python side derives the
version from Rust, so no dual edit is needed.

**Parity baseline (frozen).**
`crates/core/parity_tests/perfmodel/goldens/afd_layer_times.json` and
`goldens/afd_e2e.json` (rebuilt after `9cd9e8e9` removed the previous AFD
engine-step record). Provenance: the layer-time goldens were produced by the
retired Python island at commit `9cd9e8e9^` and are frozen — the island no
longer exists, so those numbers cannot be regenerated; the per-pool spec
fixtures under `goldens/afd_specs/` regenerate byte-for-byte via
`crates/core/parity_tests/perfmodel/gen_afd_spec_fixtures.py`, and the demo
configs the e2e goldens point at are committed under
`crates/core/parity_tests/perfmodel/configs/`. Diffed by
`scripts/afd_parity.py` (layer times 1e-9 relative, e2e 1e-6 relative,
common fields) and gated by `test_afd_parity.py` plus the Rust golden tests
(`cargo test perfmodel::afd` / `scheduler::afd`). Cases: pure `both`,
`afd+pd` decode, `afd+pd` prefill on
`h200_sxm`/`trtllm`, plus a composed hetero case (`h200_sxm` A + `b200_sxm`
F) pinning per-pool DB reads. End-to-end equivalence gates run on
admission-unsaturated workloads; queueing-region divergence between the old
greedy loop and the new heap-driven runtime is a documented fidelity
improvement, re-baselined with attribution — never a gate blocker.

**Migration (completed).** The Python island is deleted, the native path
is the only path, and `AISIMULATE_AFD_NATIVE=0` fails loudly (the retired
`shadow` dual-run mode is rejected with it; `scripts/afd_parity.py`
exposes `layers` and `e2e` only).

## Terminal State ADR v2: AFD on the Continuous SGLang Core

> Status: implemented. Supersedes the "execution-model axis"
> decision above where the two conflict.

**Decision.** AFD no longer owns a scheduling core. The pool runs on the
monolithic `SglangCore` continuous-batching scheduler (mid-flight admission,
`retract_decode` retraction, radix KV) with `"aic-afd"` injected through the
standard `PerfModel::External` seam — the same seam the retired `AfdCore`
consumed, so no scheduler-loop change was needed. `engine_core` /
`EngineCoreKind` and `AfdCore` are deleted; the rank's `afd` topology stays
as pure metadata (replay GPU accounting, the memory gate's phase). The
scheduler surface is derived from the topology: `max_num_seqs =
n_a_workers * a_batch_size` (the provider's `total_batch_size`),
`num_gpu_blocks` derived by the native provider from the memory gate's own
A-pool KV budget (`estimate_afd_kv_budget_bytes` — per-GPU budget converted
to tokens, summed across A workers) so runtime retraction reflects physical
OOM (explicit `num_gpu_blocks` pins still win), and
`chunked_prefill_size`/`max_prefill_tokens` default to the prompt window so a
prompt pre-fills in one pass. An explicit `engine.afd.chunked_prefill_size`
smaller than the prompt opts into SGLang's native chunked prefill: each
chunk pass is priced by the provider at its actual (chunk tokens, prefix)
geometry — attention grows with the real prefix, F-pool cost tracks chunk
tokens — and TTFT completes at the last chunk (see the
`qwen3-32b-h200-sglang-chunked-prefill` e2e case).

**backend is single-duty.** `mode=afd` requires `backend: sglang`:
scheduling flavor and the perf-DB family the provider queries move together,
and the rebuilt goldens anchor sglang data (`0.5.14` on `h200_sxm`). One
data caveat: the `h200_sxm` sglang family lacks `mla_bmm` rows, so that op
family falls back to trtllm data with a loud load warning.

**Timing identity.** Fixed synthetic workloads keep the island's frozen
decode identity (every pass pays the stride-averaged full-batch step).
Trace/session/agentic workloads use the instantaneous per-pass evaluation
with the effective microbatch count clamped to what the actual pool-wide
in-flight batch can fill (`micro_full = ceil(a_batch_size / num_microbatches)`,
`m' = clamp(ceil(per_worker / micro_full), 1, num_microbatches)`, even
split). The clamp contract delivers pool-wide batch counts to the provider
and converts per-worker internally; `m' == 1` degrades to the serial
cadence. Pipeline `fill` is paid per pass by default (conservative);
`engine.afd.pipeline_fill_mode: amortized` opts into the steady-state
caliber: decode passes at the configured pipeline depth pay cycle terms
only, the fill is charged on the first decode pass (frozen identity) or on
boundary passes with `m' < num_microbatches` (instantaneous path).

**Retirements (fail-loud).** `backend != sglang` AFD configs are rejected at
config validation; old wires carrying `engine_core: "afd"` fail serde
deserialization.

**afd+pd.** The two-pool handoff runs on SglangCore's
generic handoff command surface (`ReserveDestination` /
`ActivateDestination` / source holds) with the AFD pool as one disagg role —
no AFD-specific scheduler hook was needed. The rank's `afd` metadata carries
`phase: prefill|decode`; config validation requires it to match the rank's
topology-stamped `worker_type` (pure AFD stays `both` + aggregated). One
deliberate caliber difference from the retired island, pinned by the
generic-disagg convention: the destination produces ALL `osl` tokens after
activation (the source-side first-token signal marks TTFT/conformance only
and is not counted in report totals), so an AFD decode role runs one more
decode pass per request than the island's `osl-1` convention, and TTFT
includes the first destination decode pass. The unified recommend input
pins companion shapes via `engine.workers.<companion>.parallelism.preset`;
the raw search surface uses `afd_companion_parallel_configs`.

**Parity baseline (rebuilt).** `goldens/afd_e2e.json` is recollected from
the continuous path: `qwen3-32b-h200-sglang-pure-both` (unsaturated and
saturated), the `qwen3-32b-h200-sglang-pd-decode` / `-pd-prefill`
afd+pd cases, `qwen3-32b-h200-sglang-chunked-prefill` (opt-in), and
`qwen3-32b-h200-sglang-fill-amortized` (opt-in); the batch-locked
records are dropped. The trtllm
`afd_layer_times.json` cases stay frozen as the island-migration anchor; a
self-anchored `qwen3-32b-h200-sglang-pure-both` case guards the sglang
perf-DB family at 1e-9 (fixtures regenerate via
`gen_afd_spec_fixtures.py`). Gates unchanged: `scripts/afd_parity.py` +
`test_afd_parity.py` + `cargo test perfmodel::afd` / `scheduler::afd`.
