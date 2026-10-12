// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `AfdForwardPassPerfModel` — the `"aic-afd"` timing provider.
//!
//! Numerical contract:
//!
//! - per-pool engines: one compiled `Engine` per pool system, constructed
//!   through [`ForwardPassPerfModel::best_available`] with the pool's
//!   parallel identity (A: `tp=tp_a, moe_tp=tp_a, moe_ep=1`; F: `tp=f_tp,
//!   moe_tp=f_moe_tp, moe_ep=f_moe_ep_size`, `f_tp = n_f_nodes *
//!   gpus_per_node`) — never a parallel construction path;
//! - Quant rule: each pool config carries ONLY the caller's
//!   explicit quant overrides; absent overrides stay `None` so the
//!   per-pool compile resolves quant fresh against the pool's own system
//!   (an inherited A-pool value would misprice the F pool, e.g. B200
//!   forced into nvfp4_wo);
//! - op-list walks use `query_context_op` / `query_generation_op` with
//!   `x_override` (uniform `batch * s` prefill, `batch` decode; no logits
//!   exception) and the runtime `prefix` threaded into BOTH phases;
//! - decode integrates along the KV length at `AFD_DECODE_STRIDE` with the
//!   pipeline `max` applied per stride sample before accumulation
//!   (`sum_i max(..) != max(sum_i ..)`);
//! - comm ingredients come from the four `afd_comm` ops queried at
//!   `x = a_micro_batch * tokens_per_req`, measurement-side
//!   `comm_overhead_factor = 1.0` (the pipeline evaluator applies the
//!   configured factor exactly once — see `engine::scheduler::afd::pipeline`).

use serde::{Deserialize, Serialize};

use crate::common::enums::CommQuantMode;
use crate::common::error::AicError;
use crate::operators::afd_comm::{
    AfdCombineOp, AfdFCollectiveOp, AfdTransferDirection, AfdTransferOp,
};
use crate::perfmodel::afd::partition::{AfdOpsPartition, partition_afd_ops};
use crate::perfmodel::engine::Engine;
use crate::perfmodel::session::{query_context_op, query_generation_op};
use std::sync::Arc;

/// Stride for sampling the KV length along the decode trace.
pub const AFD_DECODE_STRIDE: u32 = 32;

/// AFD pipeline regime. Named for the
/// degree of communication/compute overlap; the legacy `serial` /
/// `conservative` / `optimistic` strings stay accepted as deserialization
/// aliases so frozen goldens and existing configs keep parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfdPipelineModel {
    /// No overlap: every stage runs sequentially (legacy `serial`).
    #[serde(alias = "serial")]
    NoOverlap,
    /// Partial overlap: each pool blocks on its own transfer (legacy
    /// `conservative`).
    #[serde(alias = "conservative")]
    PartialOverlap,
    /// Full overlap: the transfer is its own hidden pipeline stage (legacy
    /// `optimistic`).
    #[serde(alias = "optimistic")]
    FullOverlap,
}

/// AFD phase selector for the measurement API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfdPhase {
    Prefill,
    Decode,
}

/// A-pool (attention) construction identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AfdAPoolSpec {
    pub system: String,
    pub tp_a: u32,
    pub n_a_nodes: u32,
    pub a_batch_size: u32,
}

/// F-pool (FFN/MoE) construction identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AfdFPoolSpec {
    pub system: String,
    pub f_moe_ep_size: u32,
    pub n_f_nodes: u32,
}

/// Explicit per-model quant overrides shared by both pools (the caller's
/// explicit choice, preserved verbatim per perfmodel-api). Absent fields
/// stay `None` so each pool's compile resolves quant fresh against its own
/// system — the quant rule. NEVER populate these from the other pool's
/// resolved values: an A-pool resolution inherited by the F pool misprices
/// it (e.g. B200 forced into nvfp4_wo).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AfdQuantOverrides {
    pub gemm_quant_mode: Option<String>,
    pub moe_quant_mode: Option<String>,
    pub fmha_quant_mode: Option<String>,
    pub kvcache_quant_mode: Option<String>,
    pub comm_quant_mode: Option<String>,
}

/// Estimator plumbing shared by both pool configs (perfmodel-api's
/// controls, passed through unchanged).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AfdEstimatorPlumbing {
    pub database_mode: crate::DatabaseMode,
    pub transfer_policy: Option<Vec<String>>,
    pub systems_paths: Vec<std::path::PathBuf>,
    pub estimation_mode: crate::EstimationMode,
    pub fallback_policy: crate::ForwardPassFallbackPolicy,
    pub estimator_config: crate::EstimatorConfig,
}

/// Construction input for [`AfdForwardPassPerfModel`].
///
/// `gpus_per_node` is the node width the candidate is enumerated against;
/// `build` validates it against each
/// pool's perf-DB system spec (a mismatch means the candidate was built from
/// a different SKU definition).
#[derive(Clone, Debug)]
pub struct AfdForwardPassModelConfig {
    pub model: String,
    pub backend: crate::BackendKind,
    pub backend_version: Option<String>,
    pub a_pool: AfdAPoolSpec,
    pub f_pool: AfdFPoolSpec,
    pub gpus_per_node: u32,
    pub num_microbatches: u32,
    pub pipeline_model: AfdPipelineModel,
    /// Measurement-side value is 1.0; the pipeline evaluator
    /// (`engine::scheduler::afd::pipeline`) applies the configured factor
    /// exactly once on the transfer terms.
    pub comm_overhead_factor: f64,
    pub boundary_on_attn: bool,
    pub num_layers: u32,
    pub hidden_size: u32,
    /// MoE dispatch identity of the model (0/0 for dense — the send
    /// probability degenerates to `1 / num_f_nodes`).
    pub num_experts: u32,
    pub topk: u32,
    pub comm_quant_mode: CommQuantMode,
    pub nextn: u32,
    pub quant: AfdQuantOverrides,
    pub estimator: AfdEstimatorPlumbing,
}

/// Per-phase layer ingredients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AfdLayerTimes {
    pub attention_ms: f64,
    pub ffn_ms: f64,
    pub a_to_f_ms: f64,
    pub f_to_a_ms: f64,
    pub num_layers: u32,
}

/// Full phase metrics beyond the frozen measurement (the pipeline cadence
/// the AFD scheduler core consumes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AfdPhaseMetrics {
    pub layer_times: AfdLayerTimes,
    /// Per-layer cycle cadence (decode: stride-averaged).
    pub t_cycle_ms: f64,
    /// Full pass latency: the global pipeline step
    /// (fill + cycle * (ub * layers - 1)) for BOTH phases, stride-averaged for
    /// decode. Mirrors the billed `evaluate_afd_phase` step at the
    /// measurement-side `comm_overhead_factor = 1.0`.
    pub t_step_ms: f64,
    /// FullOverlap regime only: the network stage is fully hidden by compute.
    pub comm_hidden: bool,
}

/// One pool's compiled engine plus its precomputed A/F partitions.
struct PoolRuntime {
    engine: Arc<Engine>,
    context: AfdOpsPartition,
    generation: AfdOpsPartition,
}

/// The `"aic-afd"` forward-pass perf model.
pub struct AfdForwardPassPerfModel {
    config: AfdForwardPassModelConfig,
    a_pool: PoolRuntime,
    f_pool: PoolRuntime,
    a2f: AfdTransferOp,
    f2a: AfdTransferOp,
    f_ag: AfdFCollectiveOp,
    f_rs: AfdFCollectiveOp,
    a_combine: AfdCombineOp,
}

impl AfdForwardPassPerfModel {
    /// The construction identity (topology, pipeline knobs, estimator
    /// plumbing) — the `AfdTimingModel` derives its pipeline knobs from it.
    pub(crate) fn config(&self) -> &AfdForwardPassModelConfig {
        &self.config
    }

    /// Production constructor: one `ForwardPassPerfModel::best_available`
    /// per pool system (perfmodel-api's single construction interface), then
    /// [`Self::from_engines`]. Requires the `python` feature (engine
    /// compilation crosses into Python).
    ///
    /// Per-pool identity mapping:
    /// - A pool: `tp = tp_a`, `attention_dp = 1`, `moe_tp = tp_a`,
    ///   `moe_ep = 1` (MoE dims inert but constraint-satisfying);
    /// - F pool: `tp = n_f_nodes * gpus_per_node`, `moe_tp = f_tp /
    ///   f_moe_ep_size`, `moe_ep = f_moe_ep_size` (divisibility checked).
    ///
    /// Both pools are op-level only: the A/F partition needs granular op
    /// lists, so whole-model FPM modes reject explicitly (perfmodel-api:
    /// unsupported combinations fail, never discard settings).
    #[cfg(feature = "python")]
    pub fn build(config: AfdForwardPassModelConfig) -> Result<Self, AicError> {
        // Architecture gate: reject models whose stages/residency cannot be
        // split across A/F pools before compilation turns them into
        // confusing partition errors.
        let architecture = pyo3::Python::with_gil(|py| {
            use pyo3::prelude::PyAnyMethods;
            let module = py.import("aisimulate_core.sdk.utils")?;
            let info = module
                .getattr("get_model_config_from_model_path")?
                .call1((config.model.as_str(),))?;
            let architecture: Option<String> = info.get_item("architecture")?.extract()?;
            Ok::<_, pyo3::PyErr>(architecture.unwrap_or_default())
        })
        .map_err(|error| {
            AicError::InvalidEngineConfig(format!(
                "AFD could not resolve the architecture of model {:?}: {error}",
                config.model
            ))
        })?;
        crate::perfmodel::afd::partition::validate_afd_model_architecture(&architecture)?;
        let a_config = pool_fpm_config(&config, PoolKind::A)?;
        let f_config = pool_fpm_config(&config, PoolKind::F)?;
        let a_fpm = crate::perfmodel::ForwardPassPerfModel::best_available(a_config)?;
        let f_fpm = crate::perfmodel::ForwardPassPerfModel::best_available(f_config)?;
        let a_engine = a_fpm.native_engine().ok_or_else(|| {
            AicError::InvalidEngineConfig(
                "AFD requires a native per-pool engine (op lists); the A pool resolved to a \
                 regression-only model"
                    .into(),
            )
        })?;
        let f_engine = f_fpm.native_engine().ok_or_else(|| {
            AicError::InvalidEngineConfig(
                "AFD requires a native per-pool engine (op lists); the F pool resolved to a \
                 regression-only model"
                    .into(),
            )
        })?;
        Self::from_engines(config, a_engine, f_engine)
    }

    /// Build from two already-compiled pool engines. Production callers use
    /// [`Self::build`]; this constructor is the feature-independent core the
    /// spec-fixture parity tests drive directly.
    pub fn from_engines(
        config: AfdForwardPassModelConfig,
        a_engine: Arc<Engine>,
        f_engine: Arc<Engine>,
    ) -> Result<Self, AicError> {
        // gpus_per_node must agree with each pool's DB system spec.
        for (pool, engine) in [("a", &a_engine), ("f", &f_engine)] {
            let db = engine.database();
            let db_gpus = db.system_spec.node.num_gpus_per_node;
            if db_gpus != config.gpus_per_node {
                return Err(AicError::InvalidEngineConfig(format!(
                    "AFD {pool}-pool topology gpus_per_node={} disagrees with the '{}' system \
                     spec, which reports {db_gpus}",
                    config.gpus_per_node, db.system,
                )));
            }
        }
        let a_pool = PoolRuntime {
            context: partition_afd_ops(a_engine.context_ops(), config.boundary_on_attn)?,
            generation: partition_afd_ops(a_engine.generation_ops(), config.boundary_on_attn)?,
            engine: a_engine,
        };
        let f_pool = PoolRuntime {
            context: partition_afd_ops(f_engine.context_ops(), config.boundary_on_attn)?,
            generation: partition_afd_ops(f_engine.generation_ops(), config.boundary_on_attn)?,
            engine: f_engine,
        };
        let n_a_workers = n_a_workers(&config);
        let n_f_workers = config.f_pool.n_f_nodes * config.gpus_per_node;
        let shared_transfer = |direction: AfdTransferDirection, name: &str| AfdTransferOp {
            name: name.into(),
            scale_factor: 1.0,
            direction,
            hidden_size: config.hidden_size,
            n_a_workers,
            n_f_workers,
            gpus_per_node: config.gpus_per_node,
            num_experts: config.num_experts,
            topk: config.topk,
            comm_quant_mode: config.comm_quant_mode,
            // Measurement-side ops carry 1.0: the pipeline evaluator applies
            // the configured factor exactly once.
            comm_overhead_factor: 1.0,
        };
        let shared_collective = |collective: AfdFCollectiveOp| AfdFCollectiveOp {
            hidden_size: config.hidden_size,
            n_a_workers,
            n_f_workers,
            gpus_per_node: config.gpus_per_node,
            num_experts: config.num_experts,
            topk: config.topk,
            comm_quant_mode: config.comm_quant_mode,
            ..collective
        };
        Ok(Self {
            a2f: shared_transfer(AfdTransferDirection::A2f, "afd_a2f_transfer"),
            f2a: shared_transfer(AfdTransferDirection::F2a, "afd_f2a_transfer"),
            f_ag: shared_collective(AfdFCollectiveOp::all_gather("afd_f_node_allgather", 1.0)),
            f_rs: shared_collective(AfdFCollectiveOp::reduce_scatter(
                "afd_f_node_reducescatter",
                1.0,
            )),
            a_combine: AfdCombineOp {
                name: "afd_a_side_combine".into(),
                scale_factor: 1.0,
                hidden_size: config.hidden_size,
                tp_a: config.a_pool.tp_a,
                f_moe_ep_size: config.f_pool.f_moe_ep_size,
                comm_quant_mode: config.comm_quant_mode,
            },
            config,
            a_pool,
            f_pool,
        })
    }

    /// Partition weight sum (GiB) of one pool's assigned ops for one phase
    /// list: ``sum(op.weight_bytes()) / (1 << 30)`` over the phase partition.
    /// Used by the native memory-feasibility gate.
    pub(crate) fn pool_partition_weight_gb(&self, a_pool: bool, phase: AfdPhase) -> f64 {
        let pool = if a_pool { &self.a_pool } else { &self.f_pool };
        let (partition, ops) = match phase {
            AfdPhase::Prefill => (&pool.context, pool.engine.context_ops()),
            AfdPhase::Decode => (&pool.generation, pool.engine.generation_ops()),
        };
        let indices = if a_pool {
            &partition.attn_ops
        } else {
            &partition.ffn_ops
        };
        indices
            .iter()
            .map(|&index| ops[index].weight_bytes())
            .sum::<f64>()
            / (1u64 << 30) as f64
    }

    /// Per-phase measurement: stride-integrated for decode, single-shot for
    /// prefill.
    pub fn predict_phase_metrics(
        &self,
        phase: AfdPhase,
        isl: u32,
        osl: u32,
        prefix: u32,
    ) -> Result<AfdPhaseMetrics, AicError> {
        let cfg = &self.config;
        let num_microbatches = cfg.num_microbatches.max(1);
        let a_micro_batch_size = (cfg.a_pool.a_batch_size.max(1)).div_ceil(num_microbatches);
        let b_micro_total = n_a_workers(cfg) * a_micro_batch_size;

        // Comm bills by token volume per step (s-independent): one query
        // each, outside any stride loop.
        let tokens_per_req = match phase {
            AfdPhase::Prefill => isl.saturating_sub(prefix).max(1),
            AfdPhase::Decode => cfg.nextn + 1,
        };
        let a_batch_tokens = a_micro_batch_size * tokens_per_req;
        // Pool-local DB assignment: A-side terms (cross-pool P2P legs and the
        // A-side combine) read the A pool's DB; intra-F-node collectives read
        // the F pool's DB. Homogeneous builds share one DB. The cross-SKU
        // fabric model replaces the transfer legs' A-DB reads.
        let a_db = self.a_pool.engine.database().as_ref();
        let f_db = self.f_pool.engine.database().as_ref();
        // Cross-SKU fabric: same system -> the twin reads the pool DB
        // directly; different systems -> the (a_system, f_system) link from
        // the two node specs.
        let fabric = if cfg.a_pool.system == cfg.f_pool.system {
            None
        } else {
            Some(crate::operators::afd_comm::AfdFabricLink::between(
                &a_db.system_spec.node,
                &f_db.system_spec.node,
            ))
        };
        let t_a2f = self
            .a2f
            .query_with_fabric(a_db, a_batch_tokens, fabric)?
            .latency_ms;
        let t_f2a = self
            .f2a
            .query_with_fabric(a_db, a_batch_tokens, fabric)?
            .latency_ms;
        let brk_a = self.a_combine.query(a_db, a_batch_tokens)?.latency_ms;
        let brk_f = self.f_ag.query(f_db, a_batch_tokens)?.latency_ms
            + self.f_rs.query(f_db, a_batch_tokens)?.latency_ms;

        let num_layers = cfg.num_layers.max(1);
        match phase {
            AfdPhase::Prefill => {
                let effective_prefill_len = isl.saturating_sub(prefix).max(1);
                let t_a_total = self.sum_context_ops(
                    &self.a_pool,
                    AfdPoolSide::Attn,
                    a_micro_batch_size,
                    effective_prefill_len,
                    prefix,
                )?;
                let t_f_total = self.sum_context_ops(
                    &self.f_pool,
                    AfdPoolSide::Ffn,
                    b_micro_total,
                    effective_prefill_len,
                    prefix,
                )?;
                let t_a_layer = t_a_total / f64::from(num_layers) + brk_a;
                let t_f_layer = t_f_total / f64::from(num_layers) + brk_f;
                // Same global-step assembly as decode and the billed
                // `evaluate_afd_phase`: fill plus one cycle per
                // (microbatch * layer) unit.
                let (t_step, t_cycle, comm_hidden) = self.pipeline_global_step_latency(
                    t_a_layer,
                    t_f_layer,
                    t_a2f,
                    t_f2a,
                    num_microbatches,
                );
                Ok(AfdPhaseMetrics {
                    layer_times: AfdLayerTimes {
                        attention_ms: t_a_layer,
                        ffn_ms: t_f_layer,
                        a_to_f_ms: t_a2f,
                        f_to_a_ms: t_f2a,
                        num_layers: cfg.num_layers,
                    },
                    t_cycle_ms: t_cycle,
                    t_step_ms: t_step,
                    comm_hidden,
                })
            }
            AfdPhase::Decode => self.integrate_decode(
                isl,
                osl,
                prefix,
                a_micro_batch_size,
                b_micro_total,
                brk_a,
                brk_f,
                t_a2f,
                t_f2a,
            ),
        }
    }

    /// Instantaneous decode-step latency at one KV position (per-pass replay
    /// semantics for trace/agentic workloads): the pipeline evaluation at
    /// `s` with no stride integration. The frozen-measurement path
    /// ([`Self::predict_phase_metrics`]) keeps the stride-averaged identity
    /// for fixed synthetic workloads.
    ///
    /// `batch_size` is the POOL-WIDE in-flight sequence count delivered by
    /// the scheduler (a single rank models the whole pool). The effective
    /// microbatch count is clamped to what the actual batch can fill —
    /// see [`Self::effective_microbatch_geometry`].
    ///
    /// The returned `t_step_ms` / `t_cycle_ms` are DIAGNOSTIC: they are built
    /// from the unfactored transfer terms, so they exclude
    /// `comm_overhead_factor`, and nothing outside this module reads them. The
    /// billing exit is `evaluate_afd_phase`
    /// (`engine::scheduler::afd::timing`), which applies the factor to the two
    /// transfer terms exactly once and consumes only `layer_times` from here.
    pub fn predict_decode_step_ms(
        &self,
        batch_size: u32,
        s: u32,
        prefix: u32,
    ) -> Result<AfdPhaseMetrics, AicError> {
        let cfg = &self.config;
        let (m_eff, a_micro_batch_size) = self.effective_microbatch_geometry(batch_size);
        let b_micro_total = n_a_workers(cfg) * a_micro_batch_size;
        let verify_width = cfg.nextn + 1;

        let tokens_per_req = cfg.nextn + 1;
        let a_batch_tokens = a_micro_batch_size * tokens_per_req;
        let a_db = self.a_pool.engine.database().as_ref();
        let f_db = self.f_pool.engine.database().as_ref();
        let fabric = if cfg.a_pool.system == cfg.f_pool.system {
            None
        } else {
            Some(crate::operators::afd_comm::AfdFabricLink::between(
                &a_db.system_spec.node,
                &f_db.system_spec.node,
            ))
        };
        let t_a2f = self
            .a2f
            .query_with_fabric(a_db, a_batch_tokens, fabric)?
            .latency_ms;
        let t_f2a = self
            .f2a
            .query_with_fabric(a_db, a_batch_tokens, fabric)?
            .latency_ms;
        let brk_a = self.a_combine.query(a_db, a_batch_tokens)?.latency_ms;
        let brk_f = self.f_ag.query(f_db, a_batch_tokens)?.latency_ms
            + self.f_rs.query(f_db, a_batch_tokens)?.latency_ms;

        let num_layers = cfg.num_layers.max(1);
        let t_a_step = self.sum_generation_step(
            &self.a_pool,
            AfdPoolSide::Attn,
            a_micro_batch_size * verify_width,
            s,
            prefix,
        )?;
        let t_f_step = self.sum_generation_step(
            &self.f_pool,
            AfdPoolSide::Ffn,
            b_micro_total * verify_width,
            s,
            prefix,
        )?;
        let t_a_layer = t_a_step / f64::from(num_layers) + brk_a;
        let t_f_layer = t_f_step / f64::from(num_layers) + brk_f;
        let (t_step, t_cycle, comm_hidden) =
            self.pipeline_global_step_latency(t_a_layer, t_f_layer, t_a2f, t_f2a, m_eff);
        Ok(AfdPhaseMetrics {
            layer_times: AfdLayerTimes {
                attention_ms: t_a_layer,
                ffn_ms: t_f_layer,
                a_to_f_ms: t_a2f,
                f_to_a_ms: t_f2a,
                num_layers: cfg.num_layers,
            },
            t_cycle_ms: t_cycle,
            t_step_ms: t_step,
            comm_hidden,
        })
    }

    /// Effective microbatch geometry for a pool-wide in-flight batch
    /// (continuous-batching clamp). Each A worker carries
    /// `ceil(pool / n_a_workers)` sequences; a worker at or above its
    /// configured `a_batch_size` keeps the full configured depth and prices
    /// every microbatch at `ceil(per_worker / num_microbatches)`, matching the
    /// frozen identity's convention exactly. Below saturation `m_eff`
    /// microbatches of `ceil(per_worker / m_eff)` sequences cover the
    /// per-worker batch with the same even-split `div_ceil` convention, and
    /// `m_eff == 1` degrades the pipeline evaluation to the serial cadence.
    ///
    /// The saturated case takes its own branch rather than relying on
    /// `ceil(n / ceil(n / m)) == m`: that identity only holds when `m` divides
    /// `n` (counterexample `n = 7, m = 5`: `ceil(7/5) = 2`, so
    /// `ceil(7/2) = 4 != 5`), and the clamp would then price a full batch one
    /// microbatch shallower than the frozen path does.
    pub fn effective_microbatch_geometry(&self, pool_batch_size: u32) -> (u32, u32) {
        microbatch_geometry(&self.config, pool_batch_size)
    }

    /// The frozen-measurement view of [`Self::predict_phase_metrics`].
    pub fn predict_afd_layer_times(
        &self,
        phase: AfdPhase,
        isl: u32,
        osl: u32,
        prefix: u32,
    ) -> Result<AfdLayerTimes, AicError> {
        Ok(self
            .predict_phase_metrics(phase, isl, osl, prefix)?
            .layer_times)
    }

    /// One pass over the pool's partitioned context ops with uniform
    /// `x = batch * s` (no logits exception) and the runtime prefix. The A
    /// pool sums its `attn_ops` partition, the F pool its `ffn_ops`
    /// partition.
    fn sum_context_ops(
        &self,
        pool: &PoolRuntime,
        pool_side: AfdPoolSide,
        batch_size: u32,
        seq_len: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let db = pool.engine.database().as_ref();
        let x = batch_size * seq_len;
        let indices = match pool_side {
            AfdPoolSide::Attn => &pool.context.attn_ops,
            AfdPoolSide::Ffn => &pool.context.ffn_ops,
        };
        let mut total = 0.0;
        for &index in indices {
            let op = &pool.engine.context_ops()[index];
            total +=
                query_context_op(op, db, batch_size, seq_len, prefix, 1.0, Some(x))?.latency_ms;
        }
        Ok(total)
    }

    /// One decode-step pass with `x = batch` and the runtime prefix threaded
    /// in.
    fn sum_generation_step(
        &self,
        pool: &PoolRuntime,
        pool_side: AfdPoolSide,
        batch_size: u32,
        s: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let db = pool.engine.database().as_ref();
        let indices = match pool_side {
            AfdPoolSide::Attn => &pool.generation.attn_ops,
            AfdPoolSide::Ffn => &pool.generation.ffn_ops,
        };
        let mut total = 0.0;
        for &index in indices {
            let op = &pool.engine.generation_ops()[index];
            total += query_generation_op(op, db, batch_size, 1, s, 1.0, prefix, Some(batch_size))?
                .latency_ms;
        }
        Ok(total)
    }

    /// Stride quadrature over the decode trace with the pipeline max applied
    /// per sample.
    #[allow(clippy::too_many_arguments)]
    fn integrate_decode(
        &self,
        isl: u32,
        osl: u32,
        prefix: u32,
        a_micro_batch_size: u32,
        b_micro_total: u32,
        brk_a: f64,
        brk_f: f64,
        t_a2f: f64,
        t_f2a: f64,
    ) -> Result<AfdPhaseMetrics, AicError> {
        let cfg = &self.config;
        let num_layers = cfg.num_layers.max(1);
        let verify_width = cfg.nextn + 1;

        let mut t_a_layer_sum = 0.0;
        let mut t_f_layer_sum = 0.0;
        let mut t_cycle_sum = 0.0;
        let mut t_step_sum = 0.0;
        let mut total_repeat = 0u32;
        let mut comm_hidden = false;

        let decode_steps = osl.saturating_sub(1).max(1);
        let mut i = 0;
        while i < decode_steps {
            let s_i = isl + i + 1;
            let repeat = if osl > 1 {
                AFD_DECODE_STRIDE.min(osl - 1 - i)
            } else {
                1
            };
            if repeat == 0 {
                break;
            }
            let t_a_step = self.sum_generation_step(
                &self.a_pool,
                AfdPoolSide::Attn,
                a_micro_batch_size * verify_width,
                s_i,
                prefix,
            )?;
            let t_f_step = self.sum_generation_step(
                &self.f_pool,
                AfdPoolSide::Ffn,
                b_micro_total * verify_width,
                s_i,
                prefix,
            )?;
            let t_a_layer_i = t_a_step / f64::from(num_layers) + brk_a;
            let t_f_layer_i = t_f_step / f64::from(num_layers) + brk_f;
            let (t_step_i, t_cycle_i, hidden_i) = self.pipeline_global_step_latency(
                t_a_layer_i,
                t_f_layer_i,
                t_a2f,
                t_f2a,
                self.config.num_microbatches,
            );
            comm_hidden = hidden_i;

            t_a_layer_sum += t_a_layer_i * f64::from(repeat);
            t_f_layer_sum += t_f_layer_i * f64::from(repeat);
            t_cycle_sum += t_cycle_i * f64::from(repeat);
            t_step_sum += t_step_i * f64::from(repeat);
            total_repeat += repeat;
            i += AFD_DECODE_STRIDE;
        }

        let denom = f64::from(total_repeat.max(1));
        let t_a_layer = t_a_layer_sum / denom;
        let t_f_layer = t_f_layer_sum / denom;
        Ok(AfdPhaseMetrics {
            layer_times: AfdLayerTimes {
                attention_ms: t_a_layer,
                ffn_ms: t_f_layer,
                a_to_f_ms: t_a2f,
                f_to_a_ms: t_f2a,
                num_layers: cfg.num_layers,
            },
            t_cycle_ms: t_cycle_sum / denom,
            t_step_ms: t_step_sum / denom,
            comm_hidden,
        })
    }

    /// Per-layer cadence for the three regimes, with the full_overlap ->
    /// partial_overlap degradation when the in-flight microbatches cannot
    /// fill the K=3 pipeline.
    fn pipeline_tcycle(
        &self,
        t_a: f64,
        t_f: f64,
        t_a2f: f64,
        t_f2a: f64,
        num_microbatches: u32,
    ) -> (f64, bool) {
        let num_microbatches = num_microbatches.max(1);
        let t_c = t_a2f + t_f2a;
        match self.config.pipeline_model {
            AfdPipelineModel::NoOverlap => (t_a + t_a2f + t_f + t_f2a, false),
            AfdPipelineModel::FullOverlap => {
                if num_microbatches == 1 {
                    // Single microbatch: nothing to overlap. `evaluate_afd_phase`
                    // forces the serial cadence here too, and the
                    // continuous-batching clamp makes m' == 1 the normal state
                    // at low occupancy, so this is not a misconfiguration worth
                    // warning about.
                    return (t_a + t_a2f + t_f + t_f2a, false);
                }
                let min_m = 2.0 + t_c / t_a.max(t_f).max(1e-9);
                if (num_microbatches as f64) < min_m {
                    log::warn!(
                        "AFD full_overlap pipeline: num_microbatches ({num_microbatches}) < min \
                         required ({min_m:.1}) to hide communication. Falling back to the \
                         partial_overlap model."
                    );
                    return ((t_a + t_a2f).max(t_f + t_f2a), false);
                }
                let t_cycle = t_a.max(t_f).max(t_c);
                (t_cycle, t_c <= t_a.max(t_f))
            }
            AfdPipelineModel::PartialOverlap => ((t_a + t_a2f).max(t_f + t_f2a), false),
        }
    }

    /// The global pipeline step
    /// (fill + cycle * (in-flight microbatch-layers - 1)).
    fn pipeline_global_step_latency(
        &self,
        t_a: f64,
        t_f: f64,
        t_a2f: f64,
        t_f2a: f64,
        num_microbatches: u32,
    ) -> (f64, f64, bool) {
        let num_microbatches = num_microbatches.max(1);
        let num_layers = self.config.num_layers.max(1);
        let (t_cycle, comm_hidden) = self.pipeline_tcycle(t_a, t_f, t_a2f, t_f2a, num_microbatches);
        let pipeline_fill = t_a + t_f + t_a2f + t_f2a;
        let t_global_step =
            pipeline_fill + t_cycle * f64::from((num_microbatches * num_layers).saturating_sub(1));
        (t_global_step, t_cycle, comm_hidden)
    }
}

/// One op's measured result within a phase walk (power evidence).
pub(crate) struct AfdOpEvidence {
    pub name: String,
    pub latency_ms: f64,
    pub energy_wms: f64,
    pub source: crate::operators::Source,
    /// True for the per-layer comm ops (the four AFD comm variants, queried
    /// once per layer): the caller scales them by `num_layers` when
    /// assembling pass energy. Compute ops carry their compiled per-step
    /// values already (`scale_factor` includes `num_layers`) and stay
    /// unscaled.
    pub per_layer: bool,
}

impl AfdForwardPassPerfModel {
    /// Per-op latency/energy detail for one phase (power evidence).
    ///
    /// Compute ops appear once per step (their compiled `scale_factor`
    /// already carries `num_layers`); the four comm ops appear once per
    /// layer and the caller scales them by `num_layers` when assembling
    /// pass energy. Decode folds the stride quadrature into per-op averages.
    pub(crate) fn collect_phase_operations(
        &self,
        phase: AfdPhase,
        isl: u32,
        osl: u32,
        prefix: u32,
    ) -> Result<Vec<AfdOpEvidence>, AicError> {
        let cfg = &self.config;
        let num_microbatches = cfg.num_microbatches.max(1);
        let a_micro_batch_size = (cfg.a_pool.a_batch_size.max(1)).div_ceil(num_microbatches);
        let b_micro_total = n_a_workers(cfg) * a_micro_batch_size;

        let tokens_per_req = match phase {
            AfdPhase::Prefill => isl.saturating_sub(prefix).max(1),
            AfdPhase::Decode => cfg.nextn + 1,
        };
        let a_batch_tokens = a_micro_batch_size * tokens_per_req;
        let a_db = self.a_pool.engine.database().as_ref();
        let f_db = self.f_pool.engine.database().as_ref();
        let fabric = if cfg.a_pool.system == cfg.f_pool.system {
            None
        } else {
            Some(crate::operators::afd_comm::AfdFabricLink::between(
                &a_db.system_spec.node,
                &f_db.system_spec.node,
            ))
        };

        let mut operations: Vec<AfdOpEvidence> = Vec::new();
        let mut push_comm = |name: &str, result: crate::operators::PerformanceResult| {
            operations.push(AfdOpEvidence {
                name: name.to_owned(),
                latency_ms: result.latency_ms,
                energy_wms: result.energy_wms,
                source: result.source,
                per_layer: true,
            });
        };
        push_comm(
            "afd_a2f_transfer",
            self.a2f.query_with_fabric(a_db, a_batch_tokens, fabric)?,
        );
        push_comm(
            "afd_f2a_transfer",
            self.f2a.query_with_fabric(a_db, a_batch_tokens, fabric)?,
        );
        push_comm(
            "afd_a_side_combine",
            self.a_combine.query(a_db, a_batch_tokens)?,
        );
        push_comm(
            "afd_f_node_allgather",
            self.f_ag.query(f_db, a_batch_tokens)?,
        );
        push_comm(
            "afd_f_node_reducescatter",
            self.f_rs.query(f_db, a_batch_tokens)?,
        );

        match phase {
            AfdPhase::Prefill => {
                let effective_prefill_len = isl.saturating_sub(prefix).max(1);
                self.collect_context_ops(
                    &self.a_pool,
                    AfdPoolSide::Attn,
                    &mut operations,
                    a_micro_batch_size,
                    effective_prefill_len,
                    prefix,
                )?;
                self.collect_context_ops(
                    &self.f_pool,
                    AfdPoolSide::Ffn,
                    &mut operations,
                    b_micro_total,
                    effective_prefill_len,
                    prefix,
                )?;
            }
            AfdPhase::Decode => {
                let verify_width = cfg.nextn + 1;
                let decode_steps = osl.saturating_sub(1).max(1);
                let mut folded: std::collections::BTreeMap<
                    String,
                    (f64, f64, crate::operators::Source),
                > = std::collections::BTreeMap::new();
                let mut total_repeat = 0u32;
                let mut i = 0;
                while i < decode_steps {
                    let s_i = isl + i + 1;
                    let repeat = if osl > 1 {
                        AFD_DECODE_STRIDE.min(osl - 1 - i)
                    } else {
                        1
                    };
                    if repeat == 0 {
                        break;
                    }
                    for (pool, side, batch) in [
                        (
                            &self.a_pool,
                            AfdPoolSide::Attn,
                            a_micro_batch_size * verify_width,
                        ),
                        (&self.f_pool, AfdPoolSide::Ffn, b_micro_total * verify_width),
                    ] {
                        let db = pool.engine.database().as_ref();
                        let indices = match side {
                            AfdPoolSide::Attn => &pool.generation.attn_ops,
                            AfdPoolSide::Ffn => &pool.generation.ffn_ops,
                        };
                        for &index in indices {
                            let op = &pool.engine.generation_ops()[index];
                            let result = crate::perfmodel::session::query_generation_op(
                                op,
                                db,
                                batch,
                                1,
                                s_i,
                                1.0,
                                prefix,
                                Some(batch),
                            )?;
                            let entry = folded.entry(op.name().to_owned()).or_insert((
                                0.0,
                                0.0,
                                result.source,
                            ));
                            entry.0 += result.latency_ms * f64::from(repeat);
                            entry.1 += result.energy_wms * f64::from(repeat);
                        }
                    }
                    total_repeat += repeat;
                    i += AFD_DECODE_STRIDE;
                }
                let denom = f64::from(total_repeat.max(1));
                for (name, (latency, energy, source)) in folded {
                    operations.push(AfdOpEvidence {
                        name,
                        latency_ms: latency / denom,
                        energy_wms: energy / denom,
                        source,
                        per_layer: false,
                    });
                }
            }
        }
        Ok(operations)
    }

    fn collect_context_ops(
        &self,
        pool: &PoolRuntime,
        pool_side: AfdPoolSide,
        operations: &mut Vec<AfdOpEvidence>,
        batch_size: u32,
        seq_len: u32,
        prefix: u32,
    ) -> Result<(), AicError> {
        let db = pool.engine.database().as_ref();
        let x = batch_size * seq_len;
        let indices = match pool_side {
            AfdPoolSide::Attn => &pool.context.attn_ops,
            AfdPoolSide::Ffn => &pool.context.ffn_ops,
        };
        for &index in indices {
            let op = &pool.engine.context_ops()[index];
            let result = crate::perfmodel::session::query_context_op(
                op,
                db,
                batch_size,
                seq_len,
                prefix,
                1.0,
                Some(x),
            )?;
            operations.push(AfdOpEvidence {
                name: op.name().to_owned(),
                latency_ms: result.latency_ms,
                energy_wms: result.energy_wms,
                source: result.source,
                per_layer: false,
            });
        }
        Ok(())
    }
}

/// Which pool a [`ForwardPassPerfModelConfig`] is derived for.
#[cfg(feature = "python")]
#[derive(Clone, Copy)]
enum PoolKind {
    A,
    F,
}

/// Derive one pool's [`ForwardPassPerfModelConfig`] from the AFD model
/// config: the pool's parallel identity + shared estimator plumbing +
/// the quant rule (only the caller's explicit overrides; never the
/// other pool's resolved values).
#[cfg(feature = "python")]
fn pool_fpm_config(
    config: &AfdForwardPassModelConfig,
    pool: PoolKind,
) -> Result<crate::ForwardPassPerfModelConfig, AicError> {
    use crate::perfmodel::EstimationMode;
    if matches!(
        config.estimator.estimation_mode,
        EstimationMode::FpmInterpolation | EstimationMode::FpmRegression
    ) {
        return Err(AicError::InvalidEngineConfig(
            "AFD per-pool estimation requires op_level: the A/F partition needs granular op \
             lists; whole-model FPM modes are unsupported"
                .into(),
        ));
    }
    let (tp, moe_tp, moe_ep, system) = match pool {
        PoolKind::A => (
            config.a_pool.tp_a,
            config.a_pool.tp_a,
            1,
            config.a_pool.system.clone(),
        ),
        PoolKind::F => {
            let f_tp = config.f_pool.n_f_nodes * config.gpus_per_node;
            if config.f_pool.f_moe_ep_size == 0 || f_tp % config.f_pool.f_moe_ep_size != 0 {
                return Err(AicError::InvalidEngineConfig(format!(
                    "f_moe_ep_size={} must divide f_tp_size={f_tp} (= n_f_nodes * gpus_per_node \
                     = {} * {}) so that f_moe_tp = f_tp / f_moe_ep is an integer",
                    config.f_pool.f_moe_ep_size, config.f_pool.n_f_nodes, config.gpus_per_node,
                )));
            }
            (
                f_tp,
                f_tp / config.f_pool.f_moe_ep_size,
                config.f_pool.f_moe_ep_size,
                config.f_pool.system.clone(),
            )
        }
    };
    let mut fpm_config = crate::ForwardPassPerfModelConfig::new(
        config.model.clone(),
        system,
        config.backend.clone(),
        crate::perfmodel::ForwardPassWorkerType::Aggregated,
    );
    fpm_config.backend_version = config.backend_version.clone();
    fpm_config.tp = tp;
    fpm_config.pp = 1;
    fpm_config.attention_dp = 1;
    fpm_config.moe_tp_size = Some(moe_tp);
    fpm_config.moe_ep_size = Some(moe_ep);
    // Quant rule: only the caller's explicit overrides reach the pool
    // configs; absent fields stay None for per-pool fresh resolution.
    fpm_config.gemm_quant_mode = config.quant.gemm_quant_mode.clone();
    fpm_config.moe_quant_mode = config.quant.moe_quant_mode.clone();
    fpm_config.fmha_quant_mode = config.quant.fmha_quant_mode.clone();
    fpm_config.kvcache_quant_mode = config.quant.kvcache_quant_mode.clone();
    fpm_config.comm_quant_mode = config.quant.comm_quant_mode.clone();
    fpm_config.nextn = config.nextn;
    fpm_config.database_mode = config.estimator.database_mode;
    fpm_config.transfer_policy = config.estimator.transfer_policy.clone();
    fpm_config.systems_paths = config.estimator.systems_paths.clone();
    fpm_config.estimation_mode = config.estimator.estimation_mode;
    fpm_config.fallback_policy = config.estimator.fallback_policy;
    fpm_config.estimator_config = config.estimator.estimator_config.clone();
    Ok(fpm_config)
}

/// Which pool side a generation/context walk sums (the A pool sums its
/// `attn_ops` partition, the F pool its `ffn_ops` partition).
#[derive(Clone, Copy)]
enum AfdPoolSide {
    Attn,
    Ffn,
}

/// A-node count times workers per node.
fn n_a_workers(config: &AfdForwardPassModelConfig) -> u32 {
    config.a_pool.n_a_nodes * config.gpus_per_node / config.a_pool.tp_a.max(1)
}

/// Continuous-batching microbatch geometry for a pool-wide in-flight batch:
/// `(effective_microbatches, per_worker_micro_batch)`. See
/// [`AfdForwardPassPerfModel::effective_microbatch_geometry`].
fn microbatch_geometry(config: &AfdForwardPassModelConfig, pool_batch_size: u32) -> (u32, u32) {
    let num_microbatches = config.num_microbatches.max(1);
    let a_batch_size = config.a_pool.a_batch_size.max(1);
    let per_worker = pool_batch_size.max(1).div_ceil(n_a_workers(config).max(1));
    if per_worker >= a_batch_size {
        // Saturated (or over-capacity) worker: keep the configured pipeline
        // depth. An even split leaves some microbatches smaller, but the
        // cadence is set by the LARGEST microbatch (t_cycle = max(t_a, t_f, t_c)
        // is evaluated once per stage), so every stage is priced at
        // ceil(per_worker / m). This is exactly the frozen identity's
        // convention, so a full batch reproduces it bit-for-bit.
        return (num_microbatches, per_worker.div_ceil(num_microbatches));
    }
    let micro_full = a_batch_size.div_ceil(num_microbatches);
    let m_eff = per_worker.div_ceil(micro_full).clamp(1, num_microbatches);
    (m_eff, per_worker.div_ceil(m_eff))
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    fn base_config() -> AfdForwardPassModelConfig {
        AfdForwardPassModelConfig {
            model: "Test/Model".into(),
            backend: crate::BackendKind::Trtllm,
            backend_version: None,
            a_pool: AfdAPoolSpec {
                system: "h200_sxm".into(),
                tp_a: 8,
                n_a_nodes: 1,
                a_batch_size: 18,
            },
            f_pool: AfdFPoolSpec {
                system: "h200_sxm".into(),
                f_moe_ep_size: 1,
                n_f_nodes: 1,
            },
            gpus_per_node: 8,
            num_microbatches: 3,
            pipeline_model: AfdPipelineModel::FullOverlap,
            comm_overhead_factor: 1.0,
            boundary_on_attn: true,
            num_layers: 64,
            hidden_size: 5120,
            num_experts: 0,
            topk: 0,
            comm_quant_mode: crate::common::enums::CommQuantMode::Half,
            nextn: 0,
            quant: AfdQuantOverrides::default(),
            estimator: AfdEstimatorPlumbing::default(),
        }
    }

    #[test]
    fn geometry_covers_full_partial_and_degenerate_batches() {
        let cfg = base_config();
        let geo = |pool| microbatch_geometry(&cfg, pool);
        // Full pool batch reproduces the frozen identity exactly.
        assert_eq!(geo(18), (3, 6));
        assert_eq!(geo(12), (2, 6));
        assert_eq!(geo(9), (2, 5));
        assert_eq!(geo(6), (1, 6));
        assert_eq!(geo(5), (1, 5));
        assert_eq!(geo(1), (1, 1));
        assert_eq!(geo(0), (1, 1));
        // Over-capacity: the pipeline depth still clamps at num_microbatches.
        assert_eq!(geo(36), (3, 12));
    }

    #[test]
    fn geometry_splits_pool_batch_across_a_workers() {
        let mut cfg = base_config();
        cfg.a_pool.n_a_nodes = 2;
        let geo = |pool| microbatch_geometry(&cfg, pool);
        // Two A workers: the pool-wide batch is halved per worker, so a full
        // pool matches the single-worker geometry at half the pool size.
        assert_eq!(geo(36), (3, 6));
        assert_eq!(geo(18), (2, 5));
        assert_eq!(geo(2), (1, 1));
    }

    #[test]
    fn geometry_single_microbatch_stays_serial() {
        let mut cfg = base_config();
        cfg.num_microbatches = 1;
        assert_eq!(microbatch_geometry(&cfg, 18), (1, 18));
        assert_eq!(microbatch_geometry(&cfg, 3), (1, 3));
    }

    #[test]
    fn geometry_matches_the_frozen_identity_when_not_divisible() {
        // `ceil(n / ceil(n / m)) == m` only holds when m divides n. Where it
        // does not, the clamp would price a saturated worker one microbatch
        // shallower than the frozen identity, so saturation takes its own
        // branch and keeps the configured depth.
        let mut cfg = base_config();
        cfg.num_microbatches = 7;
        // ceil(18/7) = 3, and ceil(18/3) = 6 != 7 without the short-circuit.
        assert_eq!(microbatch_geometry(&cfg, 18), (7, 3));

        let mut cfg = base_config();
        cfg.a_pool.a_batch_size = 7;
        cfg.num_microbatches = 5;
        // ceil(7/5) = 2, and ceil(7/2) = 4 != 5 without the short-circuit.
        assert_eq!(microbatch_geometry(&cfg, 7), (5, 2));
    }

    #[test]
    fn geometry_clamps_depth_below_saturation() {
        let mut cfg = base_config();
        cfg.num_microbatches = 7;
        // micro_full = ceil(18/7) = 3; below saturation the clamp still binds.
        assert_eq!(microbatch_geometry(&cfg, 17), (6, 3));
        assert_eq!(microbatch_geometry(&cfg, 3), (1, 3));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::PerfDatabase;
    use crate::perfmodel::engine::spec::EngineSpec;
    use std::path::PathBuf;

    /// Repo systems root (committed parquet data), mirroring the readiness
    /// tests' CARGO_MANIFEST_DIR-relative access.
    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aisimulate_core/systems")
    }

    fn golden_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parity_tests/perfmodel/goldens")
    }

    fn load_db(system: &str) -> Arc<PerfDatabase> {
        Arc::new(
            PerfDatabase::load(&systems_root(), system, "trtllm", "1.3.0rc20")
                .unwrap_or_else(|e| panic!("load {system} db: {e}")),
        )
    }

    fn build_engine(spec_file: &str, db: &Arc<PerfDatabase>) -> Arc<Engine> {
        let path = golden_dir().join("afd_specs").join(spec_file);
        let json = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "read {}: {e} — regenerate via gen_afd_spec_fixtures.py",
                path.display()
            )
        });
        let spec: EngineSpec =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        Arc::new(Engine::build(spec, db.clone()).expect("engine build"))
    }

    fn pure_config() -> AfdForwardPassModelConfig {
        AfdForwardPassModelConfig {
            model: "Qwen/Qwen3-32B".into(),
            backend: crate::BackendKind::Trtllm,
            backend_version: Some("1.3.0rc20".into()),
            a_pool: AfdAPoolSpec {
                system: "h200_sxm".into(),
                tp_a: 8,
                n_a_nodes: 1,
                a_batch_size: 18,
            },
            f_pool: AfdFPoolSpec {
                system: "h200_sxm".into(),
                f_moe_ep_size: 1,
                n_f_nodes: 1,
            },
            gpus_per_node: 8,
            num_microbatches: 3,
            pipeline_model: AfdPipelineModel::FullOverlap,
            comm_overhead_factor: 1.0,
            boundary_on_attn: true,
            num_layers: 64,
            hidden_size: 5120,
            num_experts: 0,
            topk: 0,
            comm_quant_mode: CommQuantMode::Half,
            nextn: 0,
            quant: AfdQuantOverrides::default(),
            estimator: AfdEstimatorPlumbing::default(),
        }
    }

    fn golden_case(case_id: &str) -> serde_json::Value {
        let path = golden_dir().join("afd_layer_times.json");
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&path).expect("read afd_layer_times.json"),
        )
        .expect("parse golden");
        fixture["cases"][case_id].clone()
    }

    fn assert_close(field: &str, actual: f64, expected: f64) {
        let delta = (actual - expected).abs();
        let limit = 1e-9 * expected.abs().max(1e-300);
        assert!(
            delta <= limit,
            "{field}: actual={actual} expected={expected} abs_delta={delta:.3e} rtol=1e-9"
        );
    }

    fn assert_phase(actual: &AfdLayerTimes, golden: &serde_json::Value, fields: &[&str]) {
        for &field in fields {
            let expected = golden[field].as_f64().expect("golden f64");
            let actual_value = match field {
                "attention_ms" => actual.attention_ms,
                "ffn_ms" => actual.ffn_ms,
                "a_to_f_ms" => actual.a_to_f_ms,
                "f_to_a_ms" => actual.f_to_a_ms,
                other => panic!("unexpected field {other}"),
            };
            assert_close(field, actual_value, expected);
        }
        let num_layers = golden["num_layers"].as_u64().expect("golden num_layers") as u32;
        assert_eq!(actual.num_layers, num_layers);
    }

    const ALL_FIELDS: &[&str] = &["attention_ms", "ffn_ms", "a_to_f_ms", "f_to_a_ms"];

    /// The pre-rename regime strings stay accepted as deserialization aliases
    /// (frozen goldens and existing configs still carry them), while the
    /// canonical names are what serialization emits.
    #[test]
    fn legacy_pipeline_model_strings_deserialize() {
        assert_eq!(
            serde_json::from_str::<AfdPipelineModel>("\"serial\"").unwrap(),
            AfdPipelineModel::NoOverlap
        );
        assert_eq!(
            serde_json::from_str::<AfdPipelineModel>("\"conservative\"").unwrap(),
            AfdPipelineModel::PartialOverlap
        );
        assert_eq!(
            serde_json::from_str::<AfdPipelineModel>("\"optimistic\"").unwrap(),
            AfdPipelineModel::FullOverlap
        );
        assert_eq!(
            serde_json::to_string(&AfdPipelineModel::FullOverlap).unwrap(),
            "\"full_overlap\""
        );
    }

    /// Homogeneous gate: the Rust AFD model over the compiled
    /// per-pool specs must reproduce the frozen golden layer times
    /// bit-for-bit (1e-9 relative).
    #[test]
    fn homogeneous_pure_matches_python_island_golden() {
        let db = load_db("h200_sxm");
        let a_engine = build_engine("pure_h200_a.json", &db);
        let f_engine = build_engine("pure_h200_f.json", &db);
        let model = AfdForwardPassPerfModel::from_engines(pure_config(), a_engine, f_engine)
            .expect("afd model");
        let golden = golden_case("qwen3-32b-h200-trtllm-pure-both");

        let prefill = model
            .predict_afd_layer_times(AfdPhase::Prefill, 1024, 128, 0)
            .expect("prefill predict");
        assert_phase(&prefill, &golden["phases"]["prefill"], ALL_FIELDS);

        let decode = model
            .predict_afd_layer_times(AfdPhase::Decode, 1024, 128, 0)
            .expect("decode predict");
        assert_phase(&decode, &golden["phases"]["decode"], ALL_FIELDS);
    }

    /// Invariant behind the continuous-batching geometry clamp: at the
    /// saturation point the per-pass evaluation and the frozen stride-averaged
    /// identity price the same step. Pins the invariant itself rather than a
    /// magic number, and uses a depth that does NOT divide `a_batch_size` —
    /// that is exactly where `ceil(n / ceil(n / m)) == m` fails and the
    /// short-circuit in `microbatch_geometry` is load-bearing.
    #[test]
    fn saturated_continuous_step_matches_the_frozen_identity() {
        let db = load_db("h200_sxm");
        let a_engine = build_engine("pure_h200_a.json", &db);
        let f_engine = build_engine("pure_h200_f.json", &db);
        let mut cfg = pure_config();
        cfg.num_microbatches = 7;
        let model =
            AfdForwardPassPerfModel::from_engines(cfg, a_engine, f_engine).expect("afd model");

        let isl = 1024;
        // n_a_workers = 1 * 8 / 8 = 1, so the saturated pool-wide batch is
        // exactly a_batch_size.
        let pool = 18;
        assert_eq!(model.effective_microbatch_geometry(pool), (7, 3));

        // osl=2 collapses integrate_decode's stride quadrature
        // (AFD_DECODE_STRIDE = 32) to a SINGLE sample at s = isl + 1, which is
        // the only point where the stride-averaged frozen identity and the
        // instantaneous per-pass evaluation are the same number.
        let frozen = model
            .predict_phase_metrics(AfdPhase::Decode, isl, 2, 0)
            .expect("frozen decode");
        let continuous = model
            .predict_decode_step_ms(pool, isl + 1, 0)
            .expect("continuous decode");
        assert_close("t_step_ms", continuous.t_step_ms, frozen.t_step_ms);
        assert_close("t_cycle_ms", continuous.t_cycle_ms, frozen.t_cycle_ms);
        assert_close(
            "attention_ms",
            continuous.layer_times.attention_ms,
            frozen.layer_times.attention_ms,
        );
        assert_close(
            "ffn_ms",
            continuous.layer_times.ffn_ms,
            frozen.layer_times.ffn_ms,
        );
    }

    /// `m_eff == 1` is the normal state at low occupancy under continuous
    /// batching, so the full_overlap regime must degrade to the no_overlap
    /// cadence quietly rather than warn about a misconfiguration. The billing
    /// exit enforces the same rule (see
    /// `no_overlap_and_single_microbatch_use_sequential_cadence` in
    /// `engine::scheduler::afd::pipeline`); this pins the measurement side to
    /// the identical closed form.
    #[test]
    fn full_overlap_single_microbatch_uses_the_sequential_cadence() {
        let db = load_db("h200_sxm");
        let a_engine = build_engine("pure_h200_a.json", &db);
        let f_engine = build_engine("pure_h200_f.json", &db);
        let mut cfg = pure_config();
        cfg.num_microbatches = 1;
        cfg.pipeline_model = AfdPipelineModel::FullOverlap;
        let model =
            AfdForwardPassPerfModel::from_engines(cfg, a_engine, f_engine).expect("afd model");

        let times = model
            .predict_afd_layer_times(AfdPhase::Decode, 1024, 128, 0)
            .expect("layer times");
        let (t_cycle, hidden) = model.pipeline_tcycle(
            times.attention_ms,
            times.ffn_ms,
            times.a_to_f_ms,
            times.f_to_a_ms,
            1,
        );
        assert!(!hidden, "a single microbatch cannot hide communication");
        // evaluate_afd_phase's NoOverlap arm: t_a + t_a2f + t_f + t_f2a, NOT
        // the full_overlap->partial_overlap degradation
        // ((t_a + t_a2f).max(t_f + t_f2a)) that the min_m fallback used to
        // produce here.
        let no_overlap = times.attention_ms + times.a_to_f_ms + times.ffn_ms + times.f_to_a_ms;
        assert_close("t_cycle", t_cycle, no_overlap);
        let partial_overlap =
            (times.attention_ms + times.a_to_f_ms).max(times.ffn_ms + times.f_to_a_ms);
        assert!(
            (t_cycle - partial_overlap).abs() > 1e-12,
            "single-microbatch cadence must not be the partial_overlap max"
        );
    }

    /// Continuous-path anchor: the SglangCore-injected AFD runtime queries
    /// the sglang perf-DB family; this case pins the native model's layer
    /// times at that identity (self-anchored data-drift guard, 1e-9
    /// relative). Regenerate the printed payload with
    /// `AFD_PRINT_SGLANG_LAYERS=1 cargo test -p aisimulate-core --lib
    /// sglang_pure_matches_committed_layer_times -- --nocapture`.
    #[test]
    fn sglang_pure_matches_committed_layer_times() {
        let db = Arc::new(
            PerfDatabase::load(&systems_root(), "h200_sxm", "sglang", "0.5.14")
                .expect("load sglang db"),
        );
        let a_engine = build_engine("pure_h200_a_sglang.json", &db);
        let f_engine = build_engine("pure_h200_f_sglang.json", &db);
        let mut config = pure_config();
        config.backend = crate::BackendKind::Sglang;
        config.backend_version = Some("0.5.14".into());
        let model =
            AfdForwardPassPerfModel::from_engines(config, a_engine, f_engine).expect("afd model");
        if std::env::var_os("AFD_PRINT_SGLANG_LAYERS").is_some() {
            for (name, phase) in [("prefill", AfdPhase::Prefill), ("decode", AfdPhase::Decode)] {
                let times = model
                    .predict_afd_layer_times(phase, 1024, 128, 0)
                    .expect("predict");
                eprintln!(
                    "\"{name}\": {{\"a_to_f_ms\": {:?}, \"attention_ms\": {:?}, \"f_to_a_ms\": {:?}, \"ffn_ms\": {:?}, \"num_layers\": {}}},",
                    times.a_to_f_ms,
                    times.attention_ms,
                    times.f_to_a_ms,
                    times.ffn_ms,
                    times.num_layers
                );
            }
        }
        let golden = golden_case("qwen3-32b-h200-sglang-pure-both");

        let prefill = model
            .predict_afd_layer_times(AfdPhase::Prefill, 1024, 128, 0)
            .expect("prefill predict");
        assert_phase(&prefill, &golden["phases"]["prefill"], ALL_FIELDS);

        let decode = model
            .predict_afd_layer_times(AfdPhase::Decode, 1024, 128, 0)
            .expect("decode predict");
        assert_phase(&decode, &golden["phases"]["decode"], ALL_FIELDS);
    }

    /// Hetero gate: with the F pool compiled against b200_sxm, the
    /// attention terms still come from the A pool's h200 DB while the FFN
    /// terms come from the F pool's b200 DB (the per-pool
    /// quant/data resolution rule).
    #[test]
    fn hetero_f_pool_reads_its_own_system_db() {
        let a_db = load_db("h200_sxm");
        let f_db = load_db("b200_sxm");
        let a_engine = build_engine("pure_h200_a.json", &a_db);
        let f_engine = build_engine("hetero_b200_f.json", &f_db);
        let model = AfdForwardPassPerfModel::from_engines(pure_config(), a_engine, f_engine)
            .expect("afd model");
        let golden = golden_case("qwen3-32b-h200a-b200f-trtllm-hetero-both");

        let prefill = model
            .predict_afd_layer_times(AfdPhase::Prefill, 1024, 128, 0)
            .expect("prefill predict");
        assert_phase(
            &prefill,
            &golden["phases"]["prefill"],
            &["attention_ms", "ffn_ms"],
        );

        let decode = model
            .predict_afd_layer_times(AfdPhase::Decode, 1024, 128, 0)
            .expect("decode predict");
        assert_phase(
            &decode,
            &golden["phases"]["decode"],
            &["attention_ms", "ffn_ms"],
        );

        // Transfers are per-system references until the cross-SKU fabric
        // model lands; both systems report the same value here (identical
        // node width), so the A-DB read matches the golden references.
        let prefill_refs = &golden["transfer_references"]["prefill"]["a_to_f_ms"];
        assert_close(
            "prefill a_to_f_ms reference",
            prefill.a_to_f_ms,
            prefill_refs["h200_sxm"].as_f64().unwrap(),
        );
        let decode_refs = &golden["transfer_references"]["decode"]["a_to_f_ms"];
        assert_close(
            "decode a_to_f_ms reference",
            decode.a_to_f_ms,
            decode_refs["h200_sxm"].as_f64().unwrap(),
        );
    }

    /// Enforcement: a topology enumerated against the wrong node
    /// width must fail loudly instead of pricing against a different SKU.
    #[test]
    fn gpus_per_node_mismatch_rejects() {
        let db = load_db("h200_sxm");
        let a_engine = build_engine("pure_h200_a.json", &db);
        let f_engine = build_engine("pure_h200_f.json", &db);
        let bad = AfdForwardPassModelConfig {
            gpus_per_node: 4,
            ..pure_config()
        };
        assert!(AfdForwardPassPerfModel::from_engines(bad, a_engine, f_engine).is_err());
    }

    fn wire_config() -> AfdTimingConfig {
        serde_json::from_value(serde_json::json!({
            "model": "Qwen/Qwen3-32B",
            "backend": "trtllm",
            "isl": 1024,
            "osl": 128,
            "gpus_per_node": 8,
            "num_layers": 64,
            "hidden_size": 5120,
            "a_pool": {"system": "h200_sxm", "tp_a": 8, "n_a_nodes": 1, "a_batch_size": 18},
            "f_pool": {"system": "h200_sxm", "f_moe_ep_size": 1, "n_f_nodes": 1},
            "num_microbatches": 3,
            "pipeline_model": "full_overlap",
        }))
        .expect("wire config")
    }

    /// The workload identity (isl/osl) seeds the frozen measurement and the
    /// stride integration — zero values must be rejected like every other
    /// positive-integer invariant instead of silently degrading the average.
    #[test]
    fn into_perf_config_rejects_zero_workload_identity() {
        assert!(wire_config().into_perf_config().is_ok());
        for (field, value) in [("isl", 0), ("osl", 0)] {
            let mut wire = wire_config();
            match field {
                "isl" => wire.isl = value,
                _ => wire.osl = value,
            }
            let error = wire
                .into_perf_config()
                .expect_err("zero identity must reject");
            assert!(
                format!("{error}").contains(field),
                "error must name {field}: {error}"
            );
        }
    }
}

fn default_comm_quant_mode() -> CommQuantMode {
    CommQuantMode::Half
}

fn default_one_f64() -> f64 {
    1.0
}

fn default_true() -> bool {
    true
}

/// Wire configuration of the `"aic-afd"` timing provider: the
/// JSON payload the compiler emits as `timing_model.config` for AFD roles.
/// Converts 1:1 into [`AfdForwardPassModelConfig`]; `isl`/`osl` additionally
/// seed the provider's frozen measurement identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AfdTimingConfig {
    pub model: String,
    pub backend: crate::BackendKind,
    #[serde(default)]
    pub backend_version: Option<String>,
    /// Measured workload point (the frozen measurement identity).
    pub isl: u32,
    pub osl: u32,
    pub gpus_per_node: u32,
    pub num_layers: u32,
    pub hidden_size: u32,
    #[serde(default)]
    pub num_experts: u32,
    #[serde(default)]
    pub topk: u32,
    #[serde(default = "default_comm_quant_mode")]
    pub comm_quant_mode: CommQuantMode,
    #[serde(default)]
    pub nextn: u32,
    /// KV window for the native memory gate: `Some` reserves
    /// this many tokens per request (the deployment's full context window);
    /// `None` reserves the actual `isl + osl`.
    #[serde(default)]
    pub max_seq_len: Option<u32>,
    /// Decode-identity choice: `true` (default) keeps the frozen
    /// stride-averaged decode step for fixed synthetic workloads; `false`
    /// (trace/agentic) evaluates the pipeline instantaneously per pass.
    #[serde(default = "default_true")]
    pub frozen_decode_identity: bool,
    pub a_pool: AfdAPoolSpec,
    pub f_pool: AfdFPoolSpec,
    pub num_microbatches: u32,
    pub pipeline_model: AfdPipelineModel,
    #[serde(default = "default_one_f64")]
    pub comm_overhead_factor: f64,
    #[serde(default = "default_true")]
    pub boundary_on_attn: bool,
    #[serde(default)]
    pub quant: AfdQuantOverrides,
    #[serde(default)]
    pub estimator: AfdEstimatorPlumbing,
}

impl AfdTimingConfig {
    /// Convert into the perf-model construction config, validating its
    /// positive-integer invariants.
    pub fn into_perf_config(&self) -> Result<AfdForwardPassModelConfig, AicError> {
        for (name, value) in [
            ("isl", self.isl),
            ("osl", self.osl),
            ("gpus_per_node", self.gpus_per_node),
            ("num_layers", self.num_layers),
            ("hidden_size", self.hidden_size),
            ("num_microbatches", self.num_microbatches),
            ("a_pool.tp_a", self.a_pool.tp_a),
            ("a_pool.n_a_nodes", self.a_pool.n_a_nodes),
            ("a_pool.a_batch_size", self.a_pool.a_batch_size),
            ("f_pool.n_f_nodes", self.f_pool.n_f_nodes),
            ("f_pool.f_moe_ep_size", self.f_pool.f_moe_ep_size),
        ] {
            if value < 1 {
                return Err(AicError::InvalidEngineConfig(format!(
                    "aic-afd timing config: {name} must be a positive integer, got {value}"
                )));
            }
        }
        if self.gpus_per_node % self.a_pool.tp_a != 0 {
            return Err(AicError::InvalidEngineConfig(format!(
                "aic-afd timing config: tp_a={} must divide gpus_per_node={}",
                self.a_pool.tp_a, self.gpus_per_node
            )));
        }
        if !(self.comm_overhead_factor.is_finite() && self.comm_overhead_factor > 0.0) {
            return Err(AicError::InvalidEngineConfig(format!(
                "aic-afd timing config: comm_overhead_factor must be positive finite, got {}",
                self.comm_overhead_factor
            )));
        }
        Ok(AfdForwardPassModelConfig {
            model: self.model.clone(),
            backend: self.backend.clone(),
            backend_version: self.backend_version.clone(),
            a_pool: self.a_pool.clone(),
            f_pool: self.f_pool.clone(),
            gpus_per_node: self.gpus_per_node,
            num_microbatches: self.num_microbatches,
            pipeline_model: self.pipeline_model,
            comm_overhead_factor: self.comm_overhead_factor,
            boundary_on_attn: self.boundary_on_attn,
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            num_experts: self.num_experts,
            topk: self.topk,
            comm_quant_mode: self.comm_quant_mode,
            nextn: self.nextn,
            quant: self.quant.clone(),
            estimator: self.estimator.clone(),
        })
    }
}
