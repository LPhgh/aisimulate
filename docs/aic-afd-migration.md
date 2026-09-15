<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIConfigurator AFD/CLI selective migration

This selectively ports seven commits of AIConfigurator AFD (Attention-FFN
Disaggregated) and CLI work onto AISimulate. The source commits fork from
`2ed278a91bc599c5149ddfcd527a20c3421b471d`, which predates the contiguous
main-branch synchronization boundary
`c8aee02f0887547a334c3d6cd192c42757e4b40e`; they are therefore a divergent
feature line, not a fast-forward of the recorded boundary. As with the
[#1563 transfer](aic-pr1563-migration.md), this is an individual feature
migration and `scripts/aic_sync.toml` retains the contiguous main-branch
boundary `c8aee02f0887547a334c3d6cd192c42757e4b40e`.

Source commits (upstream order):

| # | Source commit | Subject |
| --- | --- | --- |
| 1 | [`d19b08ce`](https://github.com/ai-dynamo/aiconfigurator/commit/d19b08ce2efeb0e76409a4d3b76f1777b6267db6) | feat(afd): support hetero hardware and backend per AFD pool |
| 2 | [`099c8524`](https://github.com/ai-dynamo/aiconfigurator/commit/099c85243c90a7f3d30c87120221d47f2dce78ce) | feat(afd): super-node fabric tiering and FastAFD-aligned knobs |
| 3 | [`0bdb0917`](https://github.com/ai-dynamo/aiconfigurator/commit/0bdb0917ba94de12e941d9c9049f1b5bfd10456b) | fix(afd): three modeling defects that inflated the F-pool cost |
| 4 | [`83936213`](https://github.com/ai-dynamo/aiconfigurator/commit/83936213e64e95d0027bf67deff4352d08ec507d) | docs(afd): bring the four Phase-2 TODOs in line with what shipped |
| 5 | [`279af669`](https://github.com/ai-dynamo/aiconfigurator/commit/279af669a3aaeb803ad6029ac99459a700ed33f7) | docs(afd): document the AFD CLI, Task fields and examples (#1427) |
| 6 | [`aa08c1b8`](https://github.com/ai-dynamo/aiconfigurator/commit/aa08c1b8e17bfeaed929833aa56811395914974d) | fix(afd): charge once-per-step ops outside the per-layer pipeline |
| 7 | [`ebefde85`](https://github.com/ai-dynamo/aiconfigurator/commit/ebefde85864972f99eff20e977dcaa1dea2092b8) | feat(cli): add GPU price-based throughput metrics |

The migrated tree matches upstream `ebefde85` byte-for-byte across every
mirrored path; tree equality was verified against the linearized migration
branch.

## Batch structure

The transfer is split into three self-consistent review batches. Batch 1 folds
each modeling fix (3, 6) together with the features it corrects (1, 2) so no
batch imports a known-wrong F-pool cost or pipeline cadence. Fix 6 is reordered
ahead of docs 4/5; the reorder is content-neutral because 6 touches only
`inference_session.py` and `test_afd_phase_completion.py`, which 4/5 do not.

| Batch | Commits | Scope |
| --- | --- | --- |
| PR-1 AFD modeling core | 1, 2, 3, 6 | Hetero pools, super-node fabric tiering, and the two modeling fixes; all Rust changes |
| PR-2 AFD documentation | 4, 5 | Docstrings, `example.yaml`, `advanced_tuning.md`, AFD CLI guide, examples test |
| PR-3 CLI price metrics | 7 | GPU price-based cost-efficiency column and its CLI tests |

## Behavior

AFD splits each layer across an attention (A) pool that owns the KV cache and an
FFN/MoE (F) pool, exchanging hidden activations per layer, and is orthogonal to
prefill/decode disaggregation. The migrated sweep enumerates node-granular
topologies `(n_a_nodes, n_f_nodes, tp_a, f_moe_ep_size, num_microbatches,
pipeline_model)`, supports super-node fabric tiering through a `span_gpus`
collective selection, and prices cross-pool A<->F transfers at the slower
endpoint. Each pool can be pinned to its own system, backend, and perf-database
version; unset pools inherit the top-level values.

Modeling corrections carried by this transfer:

- `num_microbatches < 2` uses the serial pipeline cadence instead of an
  unsustainable conservative fallback.
- The F-node AllGather/ReduceScatter pair is gated on `f_moe_tp_size` (a
  token-dimension TP group exists) rather than on fabric width, so pure expert
  parallelism no longer charges a collective that has nothing to exchange. The
  gate defaults to `0` ("caller did not say") and never affects sizing.
- The model-internal MoE dispatch is stripped from the F partition, since under
  AFD that all-to-all is the cross-pool transfer and must not be billed twice.
- Once-per-step ops (embedding, logits GEMM) are charged outside the per-layer
  pipeline max and paid once per micro-batch.

PR-3 adds a GPU-price-based cost-efficiency column: `gpu_hour_price` divides the
per-GPU throughput to report throughput per dollar.

Deployment artifact generation remains single-system: hetero AFD pools are
supported for modeling only, and `module_bridge.py` fails fast with an explicit
error when pools disagree on system or backend.

## AISimulate adaptations

- The AISimulate mirror is not byte-identical to upstream `c8aee02`: it also
  carries the [#1563 selective migration](aic-pr1563-migration.md) and the
  compatibility import facades. Each modified file was therefore reconciled with
  a three-way merge (base = `2ed278a`, ours = current AISimulate, theirs = the
  linearized AFD head) rather than a straight mirror patch. All merged files
  retained both the existing speculative-decoding/attention-lane code and the
  new AFD code.
- `cli/main.py` was the only textual conflict. Both sides added independent
  top-level helpers at the same anchor (`_add_attention_backend_argument` from
  the attention-lane work versus the `_parse_afd_*` / `_add_afd_pool_arguments`
  / `_collect_afd_pool_kwargs` family) and independent keyword parameters to the
  task-builder signature (`video_*` versus `afd_pools`). Both sides were kept.
- Upstream `docs/cli_user_guide.md` has no AISimulate mirror target; that guide
  was relocated to the repository-root
  [`docs/cli/legacy-aic-user-guide.md`](cli/legacy-aic-user-guide.md) during the
  original migration. The AFD bullet and the "AFD (Attention-FFN Disaggregated)
  mode" / "Heterogeneous AFD pools" sections were ported into that file by hand,
  with relative links rewritten to the AISimulate layout
  (`../../python/aisimulate/...`). `docs/advanced_tuning.md` maps to the mirror
  `python/aisimulate/docs/advanced_tuning.md`.
- `src/aiconfigurator/generator/module_bridge.py` is a rule-gated generator
  file. The change is an additive `elif serving_mode == "afd"` guard that fails
  fast on heterogeneous pools; it was reviewed against
  `python/aisimulate/.claude/rules/generator-development.md` (cross-module
  impact: CLI flags and the SDK/profiler bridge).
- The AFD field additions are cross-file at the Rust type level. Commit 2 added
  required fields `P2POp.span_gpus` and `NodeSpec.inter_rack_latency`, but the
  AISimulate perfmodel crate also carries `#[cfg(test)]` struct-literal
  constructors from the PR #1602 FPM-worker-role line that never existed on the
  AFD fork (`common/system_spec.rs`, `engine/spec.rs`, `operators/fpm_sol.rs`,
  `perf_database/gemm.rs`, `perf_database/moe_expert_compute.rs`,
  `perf_database/mod.rs`). Each was completed with the new field set to `None`,
  which preserves the pre-AFD "unset" semantics those tests assert. `cargo test
  -p aisimulate-core` passes.
- The AISimulate-only frozen operations contract
  (`tests/cross_package/test_single_oracle_contract.py`) requires every
  `operations/*.py` def to be declared. The three new defs
  (`AFDFAllGather.has_tp_group`, `AFDFReduceScatter.has_tp_group`,
  `AFDTransfer.span_gpus`) were added to `OPERATIONS_DEF_INVENTORY`; none
  computes a new performance value (two are existence-gate properties, one is a
  fabric-tier selector).
- Two imported AFD tests build a real `PerfDatabase` at raw version `0.5.16`,
  which predates the three-slot queryable-version regime (#1581) already present
  in AISimulate. Matching the existing convention for unlisted raw versions
  (e.g. `test_rust_engine_step.py`), both call sites now pass
  `allow_unlisted_version=True` rather than retargeting the data version.

- No `pyproject.toml`, `Cargo.toml`, `uv.lock`, `.github`, or `CODEOWNERS`
  changes are present, so no combined-manifest, workspace, active-workflow, or
  ownership adaptation was required and no new crate dependency was introduced.

## Attribution

Commits 1-6 are authored by `gehujun.ghj <gehujun.ghj@alibaba-inc.com>` and
commit 7 by `lj382851 <lj382851@alibaba-inc.com>`, both contributed to the
Apache-2.0 AIConfigurator project with DCO sign-off. Source copyright and
license headers are retained on the imported files. These are contributions to
the same upstream project already mirrored under `scripts/aic_sync.toml`, so no
separate root `THIRD_PARTY_NOTICES.md` entry is introduced beyond the existing
AIConfigurator attribution; re-run `python3 scripts/check_packaged_legal_files.py`
if packaged legal files change.

## Source paths

All source material below is licensed under Apache-2.0 by AIConfigurator and its
contributors. Source links name the exact tip revision `ebefde85`.

| Original source at the pinned revision | AISimulate path |
| --- | --- |
| `aic-core/rust/aiconfigurator-core/src/common/system_spec.rs` | `crates/core/src/perfmodel/common/system_spec.rs` |
| `aic-core/rust/aiconfigurator-core/src/operators/communication.rs` | `crates/core/src/perfmodel/operators/communication.rs` |
| `aic-core/rust/aiconfigurator-core/src/py_ops.rs` | `crates/core/src/perfmodel/py_ops.rs` |
| `aic-core/src/aiconfigurator_core/sdk/afd_partition.py` | `python/aisimulate/src/aiconfigurator_core/sdk/afd_partition.py` |
| `aic-core/src/aiconfigurator_core/sdk/config.py` | `python/aisimulate/src/aiconfigurator_core/sdk/config.py` |
| `aic-core/src/aiconfigurator_core/sdk/operations/afd_transfer.py` | `python/aisimulate/src/aiconfigurator_core/sdk/operations/afd_transfer.py` |
| `aic-core/src/aiconfigurator_core/sdk/perf_database.py` | `python/aisimulate/src/aiconfigurator_core/sdk/perf_database.py` |
| `aic-core/src/aiconfigurator_core/sdk/system_spec.py` | `python/aisimulate/src/aiconfigurator_core/sdk/system_spec.py` |
| `docs/advanced_tuning.md` | `python/aisimulate/docs/advanced_tuning.md` |
| `docs/cli_user_guide.md` | `docs/cli/legacy-aic-user-guide.md` (relocated; manual port) |
| `src/aiconfigurator/cli/api.py` | `python/aisimulate/src/aiconfigurator/cli/api.py` |
| `src/aiconfigurator/cli/example.yaml` | `python/aisimulate/src/aiconfigurator/cli/example.yaml` |
| `src/aiconfigurator/cli/exps/afd_hetero.yaml` | `python/aisimulate/src/aiconfigurator/cli/exps/afd_hetero.yaml` |
| `src/aiconfigurator/cli/main.py` | `python/aisimulate/src/aiconfigurator/cli/main.py` |
| `src/aiconfigurator/cli/report_and_save.py` | `python/aisimulate/src/aiconfigurator/cli/report_and_save.py` |
| `src/aiconfigurator/cli/utils.py` | `python/aisimulate/src/aiconfigurator/cli/utils.py` |
| `src/aiconfigurator/generator/module_bridge.py` | `python/aisimulate/src/aiconfigurator/generator/module_bridge.py` |
| `src/aiconfigurator/sdk/inference_session.py` | `python/aisimulate/src/aiconfigurator/sdk/inference_session.py` |
| `src/aiconfigurator/sdk/pareto_analysis.py` | `python/aisimulate/src/aiconfigurator/sdk/pareto_analysis.py` |
| `src/aiconfigurator/sdk/sweep.py` | `python/aisimulate/src/aiconfigurator/sdk/sweep.py` |
| `src/aiconfigurator/sdk/task_v2.py` | `python/aisimulate/src/aiconfigurator/sdk/task_v2.py` |
| `tests/unit/cli/test_afd_default_cli.py` | `python/aisimulate/tests/unit/cli/test_afd_default_cli.py` |
| `tests/unit/cli/test_afd_examples.py` | `python/aisimulate/tests/unit/cli/test_afd_examples.py` |
| `tests/unit/cli/test_afd_phase_completion.py` | `python/aisimulate/tests/unit/cli/test_afd_phase_completion.py` |
| `tests/unit/cli/test_cli_api.py` | `python/aisimulate/tests/unit/cli/test_cli_api.py` |
| `tests/unit/cli/test_cli_workflow.py` | `python/aisimulate/tests/unit/cli/test_cli_workflow.py` |
| `tests/unit/cli/test_plain_output.py` | `python/aisimulate/tests/unit/cli/test_plain_output.py` |
| `tests/unit/cli/test_utils.py` | `python/aisimulate/tests/unit/cli/test_utils.py` |
| `tests/unit/generator/test_module_bridge_system_name.py` | `python/aisimulate/tests/unit/generator/test_module_bridge_system_name.py` |
| `tests/unit/sdk/test_afd_hetero_pools.py` | `python/aisimulate/tests/unit/sdk/test_afd_hetero_pools.py` |
| `tests/unit/sdk/test_afd_moe_billing.py` | `python/aisimulate/tests/unit/sdk/test_afd_moe_billing.py` |
| `tests/unit/sdk/test_afd_once_per_step.py` | `python/aisimulate/tests/unit/sdk/test_afd_once_per_step.py` |
| `tests/unit/sdk/test_afd_parallel_lists.py` | `python/aisimulate/tests/unit/sdk/test_afd_parallel_lists.py` |
| `tests/unit/sdk/test_afd_partition.py` | `python/aisimulate/tests/unit/sdk/test_afd_partition.py` |
| `tests/unit/sdk/test_afd_pipeline_cadence.py` | `python/aisimulate/tests/unit/sdk/test_afd_pipeline_cadence.py` |
| `tests/unit/sdk/test_afd_supernode.py` | `python/aisimulate/tests/unit/sdk/test_afd_supernode.py` |
| `tests/unit/sdk/test_afd_transfer.py` | `python/aisimulate/tests/unit/sdk/test_afd_transfer.py` |
