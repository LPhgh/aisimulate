// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AFD communication ops — cross-pool P2P transfer and intra-pool collectives.
//!
//! Port of `aisimulate_core/sdk/operations/afd_transfer.py` (the module-level
//! TODO in that file anticipated exactly this flip, triggered by the Rust
//! runtime needing AFD natively). The
//! A/F topology math — send probability, per-link volumes, rank mapping —
//! now lives here; per-message latency values still come from the standard
//! comm twins (`P2POp` / `NcclOp` / `ElementwiseOp`), preserving the
//! single-oracle contract: these ops contain no table lookups of their own.
//!
//! Parity anchors against the Python originals:
//! - identical volume math, including `int()` truncation points and
//!   left-associative float evaluation order;
//! - `x` is the token count held by a single A-rank (the Python kwargs `x`),
//!   arriving as `RuntimeContext::num_tokens`;
//! - `AfdTransferOp` scales latency by `comm_overhead_factor` but never
//!   energy (the Python op multiplies `float(result)` only);
//! - zero cases return `PerformanceResult::zero()` (empirical-tagged).

use serde::{Deserialize, Serialize};

use crate::common::enums::CommQuantMode;
use crate::common::error::AicError;
use crate::operators::communication::{NcclOp, P2POp};
use crate::operators::elementwise::ElementwiseOp;
use crate::operators::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

/// Cross-pool transfer leg. Serde values match the Python `direction`
/// strings (`"a2f"` / `"f2a"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfdTransferDirection {
    A2f,
    F2a,
}

/// A-rank <-> F-rank mapping. Serde values match the Python `rank_mapping`
/// strings (`"one_to_one"` / `"broadcast"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfdRankMapping {
    OneToOne,
    Broadcast,
}

/// `comb(n, k)` in exact u128 arithmetic (Python `math.comb`), iterative so
/// every intermediate is itself a binomial coefficient. Returns `None` on
/// u128 overflow (only reachable at absurd expert/topk counts; callers fall
/// back to the f64 product ratio, <1e-12 relative error).
fn comb_exact(n: u32, k: u32) -> Option<u128> {
    let k = k.min(n.checked_sub(k)?);
    let mut result: u128 = 1;
    for i in 0..k {
        result = result.checked_mul(u128::from(n - i))?;
        result /= u128::from(i + 1);
    }
    Some(result)
}

/// `comb(a, k) / comb(b, k)` as an f64 product of per-factor ratios.
/// Mathematically identical to the exact quotient; used only when the exact
/// u128 route overflows.
fn comb_ratio_f64(a: u32, b: u32, k: u32) -> f64 {
    let mut ratio = 1.0f64;
    for i in 0..k {
        ratio *= f64::from(a - i) / f64::from(b - i);
    }
    ratio
}

/// Probability that a token must be dispatched to a given F-node.
///
/// Mirrors Python `_afd_send_prob`: MoE with expert parallelism uses
/// `1 - C(E - E/Nf, k) / C(E, k)`; dense models and degenerate configs use
/// `1 / num_f_nodes`.
pub(crate) fn afd_send_prob(num_experts: u32, topk: u32, num_f_nodes: u32) -> f64 {
    if num_experts == 0 || topk == 0 || num_f_nodes <= 1 {
        return 1.0 / f64::from(num_f_nodes.max(1));
    }
    let experts_per_node = num_experts / num_f_nodes;
    if experts_per_node == 0 {
        return 1.0 / f64::from(num_f_nodes);
    }
    let n_other = num_experts - experts_per_node;
    if topk > n_other {
        return 1.0;
    }
    let ratio = match (comb_exact(n_other, topk), comb_exact(num_experts, topk)) {
        (Some(other), Some(all)) => other as f64 / all as f64,
        _ => comb_ratio_f64(n_other, num_experts, topk),
    };
    1.0 - ratio
}

/// Zero return shared by every zero branch: Python
/// `PerformanceResult(0.0, 0.0, source="empirical")`.
fn empirical_zero() -> PerformanceResult {
    PerformanceResult::new(0.0, Source::Empirical)
}

/// `ceil(value / 2)` as u32 for a P2P twin's element count, erroring instead
/// of truncating when the volume exceeds the twin's u32 `hidden_size`
/// (unreachable for memory-feasible AFD topologies — the per-link volume
/// would imply terabytes of activations).
fn p2p_elements(per_link_bytes: u64) -> Result<u32, AicError> {
    let elements = per_link_bytes.div_ceil(2);
    u32::try_from(elements).map_err(|_| {
        AicError::InvalidEngineConfig(format!(
            "AFD per-link transfer volume {per_link_bytes} bytes exceeds the P2P twin's u32 element range"
        ))
    })
}

/// Cross-SKU fabric link for the AFD cross-pool transfer legs. Keyed on
/// `(a_system, f_system)`: the AFD model resolves it from the two pools'
/// node specs — bottleneck bandwidth
/// (`min` of the inter-node BWs) and worst-case latency (`max` of the P2P
/// latencies). `None` means "same system" and the transfer twin reads its
/// own DB spec.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AfdFabricLink {
    pub inter_node_bw: f64,
    pub p2p_latency: f64,
}

impl AfdFabricLink {
    /// Resolve the link between two pool node specs. The clamps mirror the
    /// P2P twin's (`inter_node_bw.max(1.0)`); the latency term is carried
    /// unclamped like the twin's.
    pub fn between(
        a: &crate::common::system_spec::NodeSpec,
        f: &crate::common::system_spec::NodeSpec,
    ) -> Self {
        Self {
            inter_node_bw: a.inter_node_bw.min(f.inter_node_bw).max(1.0),
            p2p_latency: a.p2p_latency.max(f.p2p_latency),
        }
    }
}

/// Unidirectional cross-pool P2P transfer (A->F **or** F->A).
///
/// Port of Python `AFDTransfer`. A-side operates in DP mode: each A-rank
/// holds its own independent tokens and sends/receives them with the full
/// `hidden_size` per token, so the per-link payload is symmetric for both
/// directions. `direction` declares which leg this instance models.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfdTransferOp {
    pub name: String,
    pub scale_factor: f64,
    pub direction: AfdTransferDirection,
    pub hidden_size: u32,
    pub n_a_workers: u32,
    pub n_f_workers: u32,
    pub gpus_per_node: u32,
    pub num_experts: u32,
    pub topk: u32,
    pub comm_quant_mode: CommQuantMode,
    /// Applied to LATENCY only, never energy (Python multiplies
    /// `float(result)`). Measurement-side instances carry 1.0 — the
    /// pipeline evaluator applies the configured factor exactly once.
    pub comm_overhead_factor: f64,
}

impl AfdTransferOp {
    fn clamped(&self) -> (u32, u32, u32) {
        // Python clamps in __init__ via max(..., 1) / max(..., 0).
        (
            self.n_a_workers.max(1),
            self.n_f_workers.max(1),
            self.gpus_per_node.max(1),
        )
    }

    /// Physical F-node count: `ceil(n_f_workers / gpus_per_node)`, min 1.
    pub fn num_f_nodes(&self) -> u32 {
        let (_, n_f_workers, gpus_per_node) = self.clamped();
        (n_f_workers.div_ceil(gpus_per_node)).max(1)
    }

    /// `x` = tokens held by a single A-rank.
    pub fn query(&self, db: &PerfDatabase, x: u32) -> Result<PerformanceResult, AicError> {
        self.query_with_fabric(db, x, None)
    }

    /// [`Self::query`] with an optional cross-SKU fabric override. `None`
    /// composes the standard P2P twin against `db` (same-system behavior);
    /// `Some(link)` evaluates the twin's exact formula with the link's
    /// bandwidth/latency substituted (SOL and non-SOL branches mirrored).
    pub fn query_with_fabric(
        &self,
        db: &PerfDatabase,
        x: u32,
        fabric: Option<AfdFabricLink>,
    ) -> Result<PerformanceResult, AicError> {
        if x == 0 {
            return Ok(empirical_zero());
        }
        let nf = self.num_f_nodes();
        let p_send = afd_send_prob(self.num_experts, self.topk, nf);
        let bpe = self.comm_quant_mode.mapping().memory;
        // Python: int(p_send * x * hidden_size * bpe) — left-associative
        // float product, then truncation toward zero.
        let per_link_bytes = (p_send * f64::from(x) * f64::from(self.hidden_size) * bpe) as u64;
        if per_link_bytes == 0 {
            return Ok(empirical_zero());
        }
        let result = match fabric {
            None => {
                // pp_size=2 passes the twin's "pp_size=1 is a no-op" gate; a
                // single P2P link is charged regardless of actual pp depth.
                let twin = P2POp {
                    name: "afd_p2p".into(),
                    scale_factor: 1.0,
                    pp_size: 2,
                    hidden_size: p2p_elements(per_link_bytes)?,
                    seq_split: 1,
                };
                twin.query(db, 1)?
            }
            Some(link) => {
                // The twin's formula with the fabric link substituted:
                // bytes = elements * 2 (bf16), then the same SOL /
                // non-SOL branches and non-negative clamp as P2POp::query.
                let elements = p2p_elements(per_link_bytes)?;
                let bytes = f64::from(elements) * 2.0;
                use crate::common::enums::DatabaseMode;
                use crate::operators::base::SolComponents;
                let raw = match db.database_mode {
                    DatabaseMode::Sol | DatabaseMode::SolFull => PerformanceResult::sol(
                        SolComponents::new(0.0, bytes / link.inter_node_bw * 1000.0),
                    ),
                    _ => PerformanceResult::new(
                        (bytes / link.inter_node_bw + link.p2p_latency) * 1000.0,
                        Source::Empirical,
                    ),
                };
                raw.clamp_non_negative()
            }
        };
        let latency = result.latency_ms * self.comm_overhead_factor * self.scale_factor;
        Ok(PerformanceResult::with_energy(
            latency,
            result.energy_wms * self.scale_factor,
            result.source,
        ))
    }
}

/// Shared shape of the two F-node intra-node collectives (AllGather before
/// F compute, ReduceScatter after it). Both bill by token volume along the
/// token dimension and return 0 outside one-to-one rank mapping or when the
/// node holds a single F-GPU.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfdFCollectiveOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub n_a_workers: u32,
    pub n_f_workers: u32,
    pub gpus_per_node: u32,
    pub num_experts: u32,
    pub topk: u32,
    pub comm_quant_mode: CommQuantMode,
    pub rank_mapping: AfdRankMapping,
    /// NCCL operation of the twin: `"all_gather"` or `"reduce_scatter"`.
    pub operation: String,
}

impl AfdFCollectiveOp {
    /// AllGather flavor (Python `AFDFAllGather`): collects token subsets
    /// across the node's F-GPUs before F compute.
    pub fn all_gather(name: impl Into<String>, scale_factor: f64) -> Self {
        Self::with_operation(name, scale_factor, "all_gather")
    }

    /// ReduceScatter flavor (Python `AFDFReduceScatter`): scatters each
    /// A-rank's tokens onto the corresponding F-GPU after F compute.
    pub fn reduce_scatter(name: impl Into<String>, scale_factor: f64) -> Self {
        Self::with_operation(name, scale_factor, "reduce_scatter")
    }

    fn with_operation(name: impl Into<String>, scale_factor: f64, operation: &str) -> Self {
        Self {
            name: name.into(),
            scale_factor,
            hidden_size: 0,
            n_a_workers: 1,
            n_f_workers: 1,
            gpus_per_node: 8,
            num_experts: 0,
            topk: 0,
            comm_quant_mode: CommQuantMode::Half,
            rank_mapping: AfdRankMapping::OneToOne,
            operation: operation.into(),
        }
    }

    fn clamped(&self) -> (u32, u32, u32) {
        (
            self.n_a_workers.max(1),
            self.n_f_workers.max(1),
            self.gpus_per_node.max(1),
        )
    }

    pub fn num_f_nodes(&self) -> u32 {
        let (_, n_f_workers, gpus_per_node) = self.clamped();
        (n_f_workers.div_ceil(gpus_per_node)).max(1)
    }

    /// Number of F-GPUs within a single node.
    pub fn f_gpus_in_node(&self) -> u32 {
        let (_, n_f_workers, gpus_per_node) = self.clamped();
        n_f_workers.min(gpus_per_node)
    }

    /// `x` = tokens held by a single A-rank.
    pub fn query(&self, db: &PerfDatabase, x: u32) -> Result<PerformanceResult, AicError> {
        let f_local = self.f_gpus_in_node();
        if f_local <= 1 || self.rank_mapping != AfdRankMapping::OneToOne {
            return Ok(empirical_zero());
        }
        if x == 0 {
            return Ok(empirical_zero());
        }
        let (n_a_workers, _, _) = self.clamped();
        // Python: total = x * n_a_workers (exact int); tokens_per_f_node =
        // p_send * total; per_rank_elements = int(tokens * hidden / f_local).
        let total = u64::from(x) * u64::from(n_a_workers);
        let nf = self.num_f_nodes();
        let p_send = afd_send_prob(self.num_experts, self.topk, nf);
        let tokens_per_f_node = p_send * total as f64;
        let per_rank_elements =
            (tokens_per_f_node * f64::from(self.hidden_size) / f64::from(f_local)) as u64;
        if per_rank_elements == 0 {
            return Ok(empirical_zero());
        }
        let twin = NcclOp {
            name: format!("afd_{}", self.operation),
            scale_factor: 1.0,
            hidden_size: per_rank_elements as f64,
            num_gpus: f_local,
            dtype: self.comm_quant_mode,
            operation: self.operation.clone(),
            seq_split: 1,
        };
        let result = twin.query(db, 1)?;
        Ok(PerformanceResult::with_energy(
            result.latency_ms * self.scale_factor,
            result.energy_wms * self.scale_factor,
            result.source,
        ))
    }
}

/// F-node intra-node AllGather along the token dimension before F compute
/// (Python `AFDFAllGather`).
pub type AfdFAllGatherOp = AfdFCollectiveOp;

/// F-node intra-node ReduceScatter after F compute (Python
/// `AFDFReduceScatter`).
pub type AfdFReduceScatterOp = AfdFCollectiveOp;

/// A-side cross-EP combine: local HBM reduce-add of partial results.
///
/// Port of Python `AFDCombine`. When the F side uses expert parallelism
/// (`f_moe_ep_size > 1`), each A-rank receives `f_moe_ep_size` partial
/// results and reduces them locally. Returns 0 for dense FFN.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfdCombineOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub tp_a: u32,
    pub f_moe_ep_size: u32,
    pub comm_quant_mode: CommQuantMode,
}

impl AfdCombineOp {
    /// `x` = tokens held by a single A-rank.
    pub fn query(&self, db: &PerfDatabase, x: u32) -> Result<PerformanceResult, AicError> {
        let f_moe_ep_size = self.f_moe_ep_size.max(1);
        if f_moe_ep_size <= 1 {
            return Ok(empirical_zero());
        }
        if x == 0 {
            return Ok(empirical_zero());
        }
        let tp_a = self.tp_a.max(1);
        // Python: tokens_per_a_rank = ceil(x / tp_a) (exact int math);
        // total_bytes = int((f_moe_ep_size + 1) * tokens * hidden * bpe).
        let tokens_per_a_rank = u64::from(x).div_ceil(u64::from(tp_a));
        let bpe = self.comm_quant_mode.mapping().memory;
        let total_bytes =
            (u64::from(f_moe_ep_size) + 1) * tokens_per_a_rank * u64::from(self.hidden_size);
        let total_bytes = (total_bytes as f64 * bpe) as u64;
        if total_bytes == 0 {
            return Ok(empirical_zero());
        }
        // ElementWise twin: dim_in = ceil(total_bytes/2), dim_out = 0, so
        // bytes_per_token = 2 * ceil(total_bytes/2) — byte-exact to +1 byte.
        let twin = ElementwiseOp {
            name: "afd_combine_mem".into(),
            scale_factor: 1.0,
            bytes_per_token: 2.0 * total_bytes.div_ceil(2) as f64,
            scale_num_tokens: 1,
            seq_split: 1,
        };
        let result = twin.query(db, 1)?;
        Ok(PerformanceResult::with_energy(
            result.latency_ms * self.scale_factor,
            result.energy_wms * self.scale_factor,
            result.source,
        ))
    }
}

/// Zero-check helper for tests: every zero branch must return an
/// empirical-tagged zero (Python `PerformanceResult(0.0, 0.0,
/// source="empirical")`).
#[cfg(test)]
fn assert_empirical_zero(result: &PerformanceResult) {
    assert_eq!(result.latency_ms, 0.0);
    assert_eq!(result.energy_wms, 0.0);
    assert_eq!(result.source, Source::Empirical);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::energy_test_fixtures::{
        Col, write_energy_systems_root, write_parquet,
    };

    fn fixture_db() -> (tempfile::TempDir, PerfDatabase) {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = write_energy_systems_root(tmp.path());
        // NCCL twin table at `<data_dir>/comm/nccl/<misc.nccl_version>` —
        // the fixture YAML declares `misc.nccl_version: test`. Rows span the
        // full message-size axis so op-level interpolation never misses.
        let nccl_dir = tmp.path().join("data/comm/nccl/test");
        std::fs::create_dir_all(&nccl_dir).expect("mkdir nccl");
        let _ = data;
        write_parquet(
            &nccl_dir.join("nccl_perf.parquet"),
            &[
                Col::Str("nccl_dtype", vec!["half", "half", "half", "half"]),
                Col::Str(
                    "op_name",
                    vec![
                        "all_gather",
                        "all_gather",
                        "reduce_scatter",
                        "reduce_scatter",
                    ],
                ),
                Col::I64("num_gpus", vec![8, 8, 8, 8]),
                Col::I64("message_size", vec![1, 1_000_000_000, 1, 1_000_000_000]),
                Col::F64("latency", vec![1.0, 1000.0, 2.0, 2000.0]),
                Col::F64("power", vec![100.0, 100.0, 200.0, 200.0]),
            ],
        );
        let db = PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").expect("db must load");
        (tmp, db)
    }

    fn transfer_op(num_experts: u32, topk: u32, n_f_workers: u32) -> AfdTransferOp {
        AfdTransferOp {
            name: "t".into(),
            scale_factor: 1.0,
            direction: AfdTransferDirection::A2f,
            hidden_size: 5120,
            n_a_workers: 4,
            n_f_workers,
            gpus_per_node: 8,
            num_experts,
            topk,
            comm_quant_mode: CommQuantMode::Half,
            comm_overhead_factor: 1.0,
        }
    }

    #[test]
    fn send_prob_matches_python_reference_values() {
        // Oracle: exact integer combinatorics (Python `math.comb`) evaluated
        // offline for the same arguments — not the production formula.
        // dense / degenerate: 1 / num_f_nodes
        assert_eq!(afd_send_prob(0, 0, 4), 0.25);
        assert_eq!(afd_send_prob(128, 8, 1), 1.0);
        // MoE: E=8, topk=2, Nf=2 -> 1 - C(4,2)/C(8,2) = 1 - 6/28
        assert!((afd_send_prob(8, 2, 2) - (1.0 - 6.0 / 28.0)).abs() < 1e-15);
        // MoE: E=16, topk=4, Nf=4 -> 1 - C(12,4)/C(16,4) = 1 - 495/1820
        assert!((afd_send_prob(16, 4, 4) - (1.0 - 495.0 / 1820.0)).abs() < 1e-15);
        // topk > n_other -> 1.0 (E=8, topk=7, Nf=2: n_other=4 < 7)
        assert_eq!(afd_send_prob(8, 7, 2), 1.0);
        // experts_per_node == 0 (more nodes than experts) -> 1 / Nf
        assert_eq!(afd_send_prob(2, 1, 4), 0.25);
        // DeepSeek-V3 shape: E=256, topk=8, Nf=4 ->
        // 1 - C(192,8)/C(256,8), C(192,8)=39_510_598_291_560,
        // C(256,8)=409_663_695_276_000 (math.comb, exact).
        let expected = 1.0 - 39_510_598_291_560.0f64 / 409_663_695_276_000.0f64;
        assert!((afd_send_prob(256, 8, 4) - expected).abs() < 1e-15);
    }

    #[test]
    fn transfer_composes_p2p_twin_with_hand_computed_volume() {
        let (_tmp, db) = fixture_db();
        // Dense: nf = ceil(16/8) = 2, p_send = 1/2. x = 56 tokens,
        // hidden = 5120, bf16 (2 B): per_link_bytes = int(0.5 * 56 * 5120 * 2)
        // = 286_720 -> P2P twin elements = 143_360.
        let op = transfer_op(0, 0, 16);
        let expected_twin = P2POp {
            name: "afd_p2p".into(),
            scale_factor: 1.0,
            pp_size: 2,
            hidden_size: 143_360,
            seq_split: 1,
        };
        let twin_result = expected_twin.query(&db, 1).expect("twin query");
        assert!(
            twin_result.latency_ms > 0.0,
            "twin must answer non-vacuously"
        );
        let result = op.query(&db, 56).expect("afd query");
        assert_eq!(result.latency_ms, twin_result.latency_ms);
        assert_eq!(result.energy_wms, twin_result.energy_wms);
        assert_eq!(result.source, twin_result.source);

        // MoE: E=8, topk=2, nf=2 -> p_send = 1 - 6/28 = 22/28;
        // p_send * 56 = 44 exactly -> per_link_bytes = 44 * 5120 * 2 =
        // 450_560 -> elements 225_280.
        let moe_op = transfer_op(8, 2, 16);
        let moe_twin = P2POp {
            hidden_size: 225_280,
            ..expected_twin.clone()
        };
        let moe_result = moe_op.query(&db, 56).expect("afd moe query");
        let moe_twin_result = moe_twin.query(&db, 1).expect("moe twin query");
        assert_eq!(moe_result.latency_ms, moe_twin_result.latency_ms);
    }

    #[test]
    fn transfer_scales_latency_by_overhead_but_never_energy() {
        let (_tmp, db) = fixture_db();
        let op = AfdTransferOp {
            scale_factor: 2.0,
            comm_overhead_factor: 1.5,
            ..transfer_op(0, 0, 16)
        };
        let base = transfer_op(0, 0, 16).query(&db, 56).expect("base");
        let scaled = op.query(&db, 56).expect("scaled");
        // Python: lat = float(result) * comm_overhead_factor, then
        // PerformanceResult(lat * scale_factor, energy=result.energy *
        // scale_factor) — energy sees the scale factor only.
        assert_eq!(scaled.latency_ms, base.latency_ms * 3.0);
        assert_eq!(scaled.energy_wms, base.energy_wms * 2.0);
        assert_eq!(scaled.source, base.source);
    }

    #[test]
    fn transfer_zero_token_and_zero_volume_are_empirical_zero() {
        let (_tmp, db) = fixture_db();
        assert_empirical_zero(&transfer_op(0, 0, 16).query(&db, 0).unwrap());
        // p_send * x underflows int() to 0 bytes: x=1, dense nf=32 ->
        // int((1/32) * 1 * 5120 * 2) = int(320) > 0 — instead force
        // num_experts into the degenerate 1/Nf branch with tiny volume:
        // x=1, hidden=8, nf=8 -> int(0.125 * 1 * 8 * 2) = int(2.0) = 2 > 0;
        // truly zero: hidden=1, x=1, nf=32 -> int((1/32)*1*1*2)=int(0.0625)=0.
        let tiny = AfdTransferOp {
            hidden_size: 1,
            n_a_workers: 1,
            n_f_workers: 256,
            ..transfer_op(0, 0, 256)
        };
        assert_empirical_zero(&tiny.query(&db, 1).unwrap());
    }

    #[test]
    fn all_gather_composes_nccl_twin_with_hand_computed_elements() {
        let (_tmp, db) = fixture_db();
        // f_local = min(16, 8) = 8; total = 56 * 4 = 224; dense nf=2 ->
        // tokens_per_f_node = 112; per_rank_elements = int(112 * 5120 / 8) =
        // 71_680.
        let mut op = AfdFCollectiveOp::all_gather("g", 1.0);
        op.hidden_size = 5120;
        op.n_a_workers = 4;
        op.n_f_workers = 16;
        let expected_twin = NcclOp {
            name: "afd_all_gather".into(),
            scale_factor: 1.0,
            hidden_size: 71_680.0,
            num_gpus: 8,
            dtype: CommQuantMode::Half,
            operation: "all_gather".into(),
            seq_split: 1,
        };
        let twin_result = expected_twin.query(&db, 1).expect("twin query");
        assert!(
            twin_result.latency_ms > 0.0,
            "twin must answer non-vacuously"
        );
        let result = op.query(&db, 56).expect("ag query");
        assert_eq!(result.latency_ms, twin_result.latency_ms);
        assert_eq!(result.energy_wms, twin_result.energy_wms);
        assert_eq!(result.source, twin_result.source);
    }

    #[test]
    fn reduce_scatter_uses_reduce_scatter_operation() {
        let (_tmp, db) = fixture_db();
        let mut op = AfdFCollectiveOp::reduce_scatter("r", 1.0);
        op.hidden_size = 5120;
        op.n_a_workers = 4;
        op.n_f_workers = 16;
        let expected_twin = NcclOp {
            name: "afd_reduce_scatter".into(),
            scale_factor: 1.0,
            hidden_size: 71_680.0,
            num_gpus: 8,
            dtype: CommQuantMode::Half,
            operation: "reduce_scatter".into(),
            seq_split: 1,
        };
        let twin_result = expected_twin.query(&db, 1).expect("twin query");
        let result = op.query(&db, 56).expect("rs query");
        assert_eq!(result.latency_ms, twin_result.latency_ms);
    }

    #[test]
    fn collectives_are_zero_for_broadcast_single_gpu_or_zero_tokens() {
        let (_tmp, db) = fixture_db();
        let mut op = AfdFCollectiveOp::all_gather("g", 1.0);
        op.hidden_size = 5120;
        op.n_a_workers = 4;
        op.n_f_workers = 16;
        assert_empirical_zero(&op.query(&db, 0).unwrap());
        op.rank_mapping = AfdRankMapping::Broadcast;
        assert_empirical_zero(&op.query(&db, 56).unwrap());
        op.rank_mapping = AfdRankMapping::OneToOne;
        op.n_f_workers = 1; // f_local = 1
        assert_empirical_zero(&op.query(&db, 56).unwrap());
    }

    #[test]
    fn fabric_override_substitutes_link_and_matches_twin_when_equal() {
        let (_tmp, db) = fixture_db();
        let op = transfer_op(0, 0, 16);
        let base = op.query(&db, 56).expect("base");
        // Link identical to the fixture spec (inter_node_bw=50e9,
        // p2p_latency=2e-6): the override path must reproduce the twin
        // bit-exactly (the same-SKU degradation regression).
        let same = op
            .query_with_fabric(
                &db,
                56,
                Some(AfdFabricLink {
                    inter_node_bw: 50e9,
                    p2p_latency: 2e-6,
                }),
            )
            .expect("same-link");
        assert_eq!(base.latency_ms, same.latency_ms);
        assert_eq!(base.source, same.source);
        // Hand-computed oracle for a weaker link: per_link_bytes = 286_720
        // (dense nf=2, x=56, hidden=5120, bf16) -> bytes 286_720 after the
        // ceil-even round-trip; (bytes / 25e9 + 4e-6) * 1000 ms.
        let weaker = op
            .query_with_fabric(
                &db,
                56,
                Some(AfdFabricLink {
                    inter_node_bw: 25e9,
                    p2p_latency: 4e-6,
                }),
            )
            .expect("weaker-link");
        let expected = (286_720.0 / 25e9 + 4e-6) * 1000.0;
        assert!(
            (weaker.latency_ms - expected).abs() < 1e-12 * expected,
            "weaker link: actual={} expected={expected}",
            weaker.latency_ms
        );
    }

    #[test]
    fn fabric_between_takes_bottleneck_bw_and_worst_latency() {
        use crate::common::system_spec::NodeSpec;
        let node = |bw: f64, latency: f64| NodeSpec {
            num_gpus_per_node: 8,
            inter_node_bw: bw,
            intra_node_bw: 900e9,
            pcie_bw: None,
            p2p_latency: latency,
            num_gpus_per_rack: None,
            inter_rack_bw: None,
        };
        let link = AfdFabricLink::between(&node(50e9, 2e-6), &node(25e9, 4e-6));
        assert_eq!(link.inter_node_bw, 25e9);
        assert_eq!(link.p2p_latency, 4e-6);
        // Degenerate specs are clamped like the twin (bw.max(1.0)).
        let clamped = AfdFabricLink::between(&node(0.0, 0.0), &node(0.0, 0.0));
        assert_eq!(clamped.inter_node_bw, 1.0);
    }

    #[test]
    fn combine_composes_elementwise_twin_with_hand_computed_bytes() {
        let (_tmp, db) = fixture_db();
        // ep=4, tp_a=8, x=56 -> tokens_per_a_rank = 7; total_bytes =
        // int(5 * 7 * 5120 * 2.0) = 358_400 -> twin bytes_per_token =
        // 2 * ceil(358_400 / 2) = 358_400.
        let op = AfdCombineOp {
            name: "c".into(),
            scale_factor: 1.0,
            hidden_size: 5120,
            tp_a: 8,
            f_moe_ep_size: 4,
            comm_quant_mode: CommQuantMode::Half,
        };
        let expected_twin = ElementwiseOp {
            name: "afd_combine_mem".into(),
            scale_factor: 1.0,
            bytes_per_token: 358_400.0,
            scale_num_tokens: 1,
            seq_split: 1,
        };
        let twin_result = expected_twin.query(&db, 1).expect("twin query");
        assert!(
            twin_result.latency_ms > 0.0,
            "twin must answer non-vacuously"
        );
        let result = op.query(&db, 56).expect("combine query");
        assert_eq!(result.latency_ms, twin_result.latency_ms);
        assert_eq!(result.energy_wms, twin_result.energy_wms);
        assert_eq!(result.source, twin_result.source);
    }

    #[test]
    fn combine_is_zero_without_expert_parallelism_or_tokens() {
        let (_tmp, db) = fixture_db();
        let dense = AfdCombineOp {
            name: "c".into(),
            scale_factor: 1.0,
            hidden_size: 5120,
            tp_a: 8,
            f_moe_ep_size: 1,
            comm_quant_mode: CommQuantMode::Half,
        };
        assert_empirical_zero(&dense.query(&db, 56).unwrap());
        let ep = AfdCombineOp {
            f_moe_ep_size: 4,
            ..dense
        };
        assert_empirical_zero(&ep.query(&db, 0).unwrap());
    }
}
