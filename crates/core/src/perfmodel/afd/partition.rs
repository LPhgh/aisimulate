// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AFD model operation partitioning.
//!
//! Port of `aisimulate_core/sdk/afd_partition.py`: splits a model's
//! context/generation op list into the A-worker (attention) pool and the
//! F-worker (FFN/MoE) pool. The Python classifier is isinstance/name-marker
//! based; here the judgment is **Op-variant first, name-marker fallback**,
//! with identical outcomes:
//!
//! - `CustomAllReduce` / `Nccl` / `P2P` variants are skip (the Python
//!   isinstance check), as are the name markers (`_p2p`, `_ar`,
//!   `all_reduce`, `tp_all_gather`, `dispatch`, ...);
//! - `Mamba2` / `Gdn` variants and the `mamba` / `gdn` name markers reject
//!   (unclassifiable layer families);
//! - `Overlap` recurses into its inner ops: a unanimous non-skip inner vote
//!   wins, an all-skip vote goes to `unknown_or_default` (strict reject —
//!   Python parity, NOT the op's own name markers), a mixed vote rejects
//!   (spans A/F boundaries);
//! - everything else classifies by the same boundary -> attn -> ffn marker
//!   order as Python (marker overlap makes the order significant, e.g.
//!   `proj_gemm` vs `down_proj_gemm`).

use crate::common::error::AicError;
use crate::operators::Op;

/// Pool assignment of one op (Python `AFDSide`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfdSide {
    Attn,
    Ffn,
    Boundary,
    Skip,
}

/// Partitioned index view of one model op list.
///
/// Mirrors Python `AFDOpsPartition`: `boundary` ops are ALSO present in
/// `attn_ops` or `ffn_ops` according to `boundary_on_attn`; `skipped_ops`
/// records model-internal communication/dispatch ops that AFD models outside
/// the original op list. Entries are indices into the source slice.
#[derive(Clone, Debug, Default)]
pub struct AfdOpsPartition {
    pub attn_ops: Vec<usize>,
    pub ffn_ops: Vec<usize>,
    pub boundary_ops: Vec<usize>,
    pub skipped_ops: Vec<usize>,
}

/// Partition `ops` into A/F pools. Strict: unclassified ops reject
/// (Python `allow_unknown_ops=False`, the only mode the measurement
/// path uses).
pub fn partition_afd_ops(ops: &[Op], boundary_on_attn: bool) -> Result<AfdOpsPartition, AicError> {
    partition_afd_ops_with(ops, boundary_on_attn, false, AfdSide::Attn)
}

/// Full-knob variant (Python `allow_unknown_ops` / `unknown_side`).
pub fn partition_afd_ops_with(
    ops: &[Op],
    boundary_on_attn: bool,
    allow_unknown_ops: bool,
    unknown_side: AfdSide,
) -> Result<AfdOpsPartition, AicError> {
    let mut partition = AfdOpsPartition::default();
    for (index, op) in ops.iter().enumerate() {
        match classify_op(op, allow_unknown_ops, unknown_side)? {
            AfdSide::Attn => partition.attn_ops.push(index),
            AfdSide::Ffn => partition.ffn_ops.push(index),
            AfdSide::Boundary => {
                partition.boundary_ops.push(index);
                if boundary_on_attn {
                    partition.attn_ops.push(index);
                } else {
                    partition.ffn_ops.push(index);
                }
            }
            AfdSide::Skip => partition.skipped_ops.push(index),
        }
    }
    Ok(partition)
}

/// Reject architectures whose stages/residency cannot be split across A/F
/// pools (Python `validate_afd_model_architecture`).
pub fn validate_afd_model_architecture(architecture: &str) -> Result<(), AicError> {
    if architecture == "DeepseekV41ForCausalLM" {
        return Err(AicError::UnsupportedModel(
            "AFD does not support DeepSeek-V4.1: mixed attention/FFN stages and their weight, \
             scale, Engram, and cache residency are not partitioned. Use agg, disagg, or static \
             estimation."
                .into(),
        ));
    }
    Ok(())
}

fn classify_op(
    op: &Op,
    allow_unknown_ops: bool,
    unknown_side: AfdSide,
) -> Result<AfdSide, AicError> {
    if let Op::Overlap(overlap) = op {
        return classify_overlap(op, overlap, allow_unknown_ops, unknown_side);
    }
    classify_by_markers(op, allow_unknown_ops, unknown_side)
}

fn classify_overlap(
    op: &Op,
    overlap: &crate::operators::op::OverlapOp,
    allow_unknown_ops: bool,
    unknown_side: AfdSide,
) -> Result<AfdSide, AicError> {
    let inner: Vec<&Op> = overlap.group_a.iter().chain(&overlap.group_b).collect();
    if !inner.is_empty() {
        let mut votes: Vec<AfdSide> = Vec::with_capacity(inner.len());
        for inner_op in inner {
            votes.push(classify_op(inner_op, allow_unknown_ops, unknown_side)?);
        }
        votes.retain(|side| *side != AfdSide::Skip);
        votes.dedup();
        match votes.as_slice() {
            [only] => return Ok(*only),
            [] => return unknown_or_default(op, allow_unknown_ops, unknown_side),
            _ => {
                return Err(AicError::InvalidEngineConfig(format!(
                    "OverlapOp '{}' spans A/F boundaries and cannot be kept atomic.",
                    op.name()
                )));
            }
        }
    }
    classify_by_markers(op, allow_unknown_ops, unknown_side)
}

/// Single-source marker dispatch (Python `_classify_by_markers`): the
/// skip -> unclassifiable -> boundary -> attn -> ffn check order is
/// load-bearing and must stay identical to the Python callsite order.
fn classify_by_markers(
    op: &Op,
    allow_unknown_ops: bool,
    unknown_side: AfdSide,
) -> Result<AfdSide, AicError> {
    let name = op.name().to_lowercase();
    if is_skipped_model_internal_op(op, &name) {
        return Ok(AfdSide::Skip);
    }
    raise_if_unclassifiable_layer_family(op, &name)?;
    if is_boundary_op(&name) {
        return Ok(AfdSide::Boundary);
    }
    if is_attention_side_op(&name) {
        return Ok(AfdSide::Attn);
    }
    if is_ffn_side_op(&name) {
        return Ok(AfdSide::Ffn);
    }
    unknown_or_default(op, allow_unknown_ops, unknown_side)
}

fn unknown_or_default(op: &Op, allow: bool, side: AfdSide) -> Result<AfdSide, AicError> {
    if allow {
        return Ok(side);
    }
    Err(AicError::InvalidEngineConfig(format!(
        "Cannot classify op '{}' for AFD partitioning.",
        op.name().to_lowercase()
    )))
}

fn is_skipped_model_internal_op(op: &Op, name: &str) -> bool {
    if matches!(op, Op::CustomAllReduce(_) | Op::P2P(_) | Op::Nccl(_)) {
        return true;
    }
    const SKIP_MARKERS: &[&str] = &[
        "_p2p",
        "_ar",
        "all_reduce",
        "allreduce",
        "tp_all_gather",
        "tp_reduce_scatter",
        "dispatch",
    ];
    SKIP_MARKERS.iter().any(|marker| name.contains(marker))
}

fn raise_if_unclassifiable_layer_family(op: &Op, name: &str) -> Result<(), AicError> {
    // Variant-first: the Mamba2/Gdn op families reject by type; the name
    // markers catch Python-era ops that only carry the family in their name.
    let variant_family = match op {
        Op::Mamba2(_) => Some("Mamba"),
        Op::Gdn(_) => Some("GDN"),
        _ => None,
    };
    let family = variant_family.or_else(|| {
        if name.contains("mamba") {
            Some("Mamba")
        } else if name.contains("gdn") {
            Some("GDN")
        } else {
            None
        }
    });
    if let Some(family) = family {
        return Err(AicError::InvalidEngineConfig(format!(
            "AFD op partitioner cannot safely classify {family} op '{name}'. {family} layers are \
             not covered by the current attention/FFN partition rules. Add an explicit AFD \
             partitioning rule before running estimate-mode afd for this model."
        )));
    }
    Ok(())
}

fn is_boundary_op(name: &str) -> bool {
    const BOUNDARY_MARKERS: &[&str] = &[
        "logits_gemm",
        "reduce_add",
        "add_norm_2",
        "_ffn_norm",
        "_moe_norm",
        "_mlp_norm",
        "_dense_ffn_norm",
    ];
    BOUNDARY_MARKERS.iter().any(|marker| name.contains(marker)) || name.ends_with("_combine")
}

/// Names that look like FFN GEMMs but contain `proj_gemm` as a substring
/// (Python `_FFN_PROJ_GEMM_MARKERS`): without this guard the bare
/// `proj_gemm` attention marker would pull HF-style FFN names into the
/// A-pool.
const FFN_PROJ_GEMM_MARKERS: &[&str] = &["down_proj_gemm", "up_proj_gemm", "gate_proj_gemm"];

fn is_attention_side_op(name: &str) -> bool {
    const ATTN_MARKERS: &[&str] = &[
        "embedding",
        "add_norm_1",
        "_attn_norm",
        "attention",
        "qkv",
        "q_a_layernorm",
        "q_b_proj",
        "kv_b_proj",
        "downscale_gemm",
        "mla",
        "bmm",
        "rope",
    ];
    if ATTN_MARKERS.iter().any(|marker| name.contains(marker)) {
        return true;
    }
    name.contains("proj_gemm") && !FFN_PROJ_GEMM_MARKERS.iter().any(|ffn| name.contains(ffn))
}

fn is_ffn_side_op(name: &str) -> bool {
    const FFN_MARKERS: &[&str] = &[
        "router",
        "moe",
        "ffn",
        "mlp",
        "shared",
        "expert",
        "act_gate",
        "_act",
        "gate_up",
        "gate_ffn",
        "up_gemm",
        "down_gemm",
        // HF-style FFN GEMM names, kept in sync with
        // FFN_PROJ_GEMM_MARKERS.
        "down_proj",
        "up_proj",
        "gate_proj",
        "relu",
    ];
    FFN_MARKERS.iter().any(|marker| name.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::op::OverlapOp;
    use crate::operators::{
        ContextAttentionOp, CustomAllReduceOp, GemmOp, GenerationAttentionOp, Mamba2Op, MoeOp,
        NcclOp, P2POp,
    };

    fn gemm_named(name: &str) -> Op {
        Op::Gemm(GemmOp::new(
            name,
            1024,
            1024,
            crate::common::enums::GemmQuantMode::Bfloat16,
        ))
    }

    fn side_of(op: &Op) -> AfdSide {
        classify_op(op, false, AfdSide::Attn).expect("classify")
    }

    #[test]
    fn marker_classification_matches_python_rules() {
        // Oracle: the Python marker tables in
        // aisimulate_core/sdk/afd_partition.py applied by hand to each name.
        let attn_names = [
            "embedding",
            "add_norm_1",
            "decoder_attn_norm",
            "context_attention",
            "generation_attention",
            "qkv_gemm",
            "q_a_layernorm",
            "q_b_proj_gemm",
            "kv_b_proj",
            "downscale_gemm",
            "context_mla_module",
            "context_bmm",
            "rope",
            "proj_gemm", // W_O output projection
        ];
        for name in attn_names {
            assert_eq!(side_of(&gemm_named(name)), AfdSide::Attn, "{name}");
        }
        let ffn_names = [
            "router_gemm",
            "moe",
            "ffn",
            "mlp",
            "shared_expert",
            "act_gate",
            "silu_act",
            "gate_up_gemm",
            "gate_ffn_gemm",
            "up_gemm",
            "down_gemm",
            "down_proj_gemm", // HF-style FFN names beat the proj_gemm attn marker
            "up_proj_gemm",
            "gate_proj_gemm",
            "relu",
        ];
        for name in ffn_names {
            assert_eq!(side_of(&gemm_named(name)), AfdSide::Ffn, "{name}");
        }
        let boundary_names = [
            "logits_gemm",
            "reduce_add",
            "add_norm_2",
            "decoder_ffn_norm",
            "decoder_moe_norm",
            "decoder_mlp_norm",
            "decoder_dense_ffn_norm",
            "moe_combine",
        ];
        for name in boundary_names {
            assert_eq!(side_of(&gemm_named(name)), AfdSide::Boundary, "{name}");
        }
        let skip_names = [
            "attn_p2p",
            "mlp_ar",
            "tp_all_reduce",
            "allreduce",
            "tp_all_gather",
            "tp_reduce_scatter",
            "moe_dispatch",
        ];
        for name in skip_names {
            assert_eq!(side_of(&gemm_named(name)), AfdSide::Skip, "{name}");
        }
    }

    #[test]
    fn comm_variants_skip_by_type_regardless_of_name() {
        // Python isinstance check: CustomAllReduce/P2P/NCCL skip even with
        // attention-looking names.
        let p2p = Op::P2P(P2POp::new("attention_link", 2, 5120));
        assert_eq!(side_of(&p2p), AfdSide::Skip);
        let nccl = Op::Nccl(NcclOp::new(
            "attention_all_gather",
            1.0,
            5120.0,
            8,
            "all_gather",
        ));
        assert_eq!(side_of(&nccl), AfdSide::Skip);
    }

    #[test]
    fn mamba_and_gdn_reject() {
        let mamba_named = gemm_named("mamba2_mixer");
        assert!(classify_op(&mamba_named, false, AfdSide::Attn).is_err());
        let gdn_named = gemm_named("gdn_attn");
        assert!(classify_op(&gdn_named, false, AfdSide::Attn).is_err());
    }

    #[test]
    fn unknown_ops_reject_unless_allowed() {
        let mystery = gemm_named("mystery_kernel");
        assert!(classify_op(&mystery, false, AfdSide::Attn).is_err());
        assert_eq!(
            classify_op(&mystery, true, AfdSide::Ffn).unwrap(),
            AfdSide::Ffn
        );
    }

    #[test]
    fn overlap_votes_and_mixed_overlap_rejects() {
        let attn_leaf = gemm_named("context_attention");
        let ffn_leaf = gemm_named("ffn_gemm");
        let skip_leaf = Op::P2P(P2POp::new("link", 2, 5120));

        // Unanimous non-skip vote wins (skip votes discarded).
        let unanimous = Op::Overlap(OverlapOp::new(
            "ov",
            vec![attn_leaf.clone()],
            vec![skip_leaf.clone()],
        ));
        assert_eq!(side_of(&unanimous), AfdSide::Attn);

        // Mixed A/F vote rejects.
        let mixed = Op::Overlap(OverlapOp::new("ov", vec![attn_leaf], vec![ffn_leaf]));
        assert!(classify_op(&mixed, false, AfdSide::Attn).is_err());

        // All-skip inner ops discard every vote: Python falls to
        // `_unknown_or_default` (strict reject; the default side when
        // unknown ops are allowed) — NOT to the overlap's own name markers.
        let all_skip = Op::Overlap(OverlapOp::new(
            "generation_attention_overlap",
            vec![skip_leaf.clone()],
            vec![skip_leaf],
        ));
        assert!(classify_op(&all_skip, false, AfdSide::Attn).is_err());
        assert_eq!(
            classify_op(&all_skip, true, AfdSide::Ffn).unwrap(),
            AfdSide::Ffn
        );

        // Empty overlap classifies by name markers.
        let empty = Op::Overlap(OverlapOp::new("moe_overlap", vec![], vec![]));
        assert_eq!(side_of(&empty), AfdSide::Ffn);
    }

    #[test]
    fn boundary_ops_dual_assign_per_boundary_flag() {
        let ops = vec![
            gemm_named("context_attention"),
            gemm_named("add_norm_2"),
            gemm_named("ffn_gemm"),
        ];
        let on_attn = partition_afd_ops(&ops, true).unwrap();
        assert_eq!(on_attn.attn_ops, vec![0, 1]);
        assert_eq!(on_attn.ffn_ops, vec![2]);
        assert_eq!(on_attn.boundary_ops, vec![1]);
        let on_ffn = partition_afd_ops(&ops, false).unwrap();
        assert_eq!(on_ffn.attn_ops, vec![0]);
        assert_eq!(on_ffn.ffn_ops, vec![1, 2]);
        assert_eq!(on_ffn.boundary_ops, vec![1]);
    }

    #[test]
    fn deepseek_v41_architecture_rejects() {
        assert!(validate_afd_model_architecture("DeepseekV41ForCausalLM").is_err());
        assert!(validate_afd_model_architecture("Qwen3ForCausalLM").is_ok());
    }

    /// Qwen/Qwen3-32B (h200_sxm/trtllm) op-list oracle, dumped at commit
    /// `9cd9e8e9^` (the dump script was never committed — this table is the
    /// frozen, non-regenerable record):
    /// the a_model context list is
    /// [embedding, add_norm_1, qkv_gemm, context_attention, proj_gemm,
    ///  add_norm_2, gate_ffn1_gemm, act_gate, ffn2_gemm, logits_gemm,
    ///  embedding_ar, ar_1, ar_2, p2p]
    /// with classifier sides
    /// [attn, attn, attn, attn, attn, boundary, ffn, ffn, ffn, boundary,
    ///  skip, skip, skip, skip].
    /// The generation list is identical except `generation_attention`.
    #[test]
    fn qwen3_32b_op_list_matches_python_partition_oracle() {
        use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
        use crate::operators::{ElementwiseOp, EmbeddingOp};

        let context_ops = vec![
            Op::Embedding(EmbeddingOp::new(
                "context_embedding",
                151_936,
                5120,
                GemmQuantMode::Bfloat16,
            )),
            Op::Elementwise(ElementwiseOp::new("context_add_norm_1", 20_480.0)),
            gemm_named("context_qkv_gemm"),
            Op::ContextAttention(ContextAttentionOp::new(
                "context_attention",
                8,
                1,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
            )),
            gemm_named("context_proj_gemm"),
            Op::Elementwise(ElementwiseOp::new("context_add_norm_2", 20_480.0)),
            gemm_named("context_gate_ffn1_gemm"),
            Op::Elementwise(ElementwiseOp::new("context_act_gate", 20_480.0)),
            gemm_named("context_ffn2_gemm"),
            gemm_named("context_logits_gemm"),
            Op::CustomAllReduce(CustomAllReduceOp::new("context_embedding_ar", 1.0, 5120, 8)),
            Op::CustomAllReduce(CustomAllReduceOp::new("context_ar_1", 1.0, 5120, 8)),
            Op::CustomAllReduce(CustomAllReduceOp::new("context_ar_2", 1.0, 5120, 8)),
            Op::P2P(P2POp::new("context_p2p", 2, 5120)),
        ];
        let partition = partition_afd_ops(&context_ops, true).unwrap();
        assert_eq!(partition.attn_ops, vec![0, 1, 2, 3, 4, 5, 9]);
        assert_eq!(partition.ffn_ops, vec![6, 7, 8]);
        assert_eq!(partition.boundary_ops, vec![5, 9]);
        assert_eq!(partition.skipped_ops, vec![10, 11, 12, 13]);

        // boundary_on_attn=false flips the dual assignment only.
        let flipped = partition_afd_ops(&context_ops, false).unwrap();
        assert_eq!(flipped.attn_ops, vec![0, 1, 2, 3, 4]);
        assert_eq!(flipped.ffn_ops, vec![5, 6, 7, 8, 9]);
        assert_eq!(flipped.boundary_ops, vec![5, 9]);
    }

    #[test]
    fn real_variant_leaves_classify_like_their_python_namesakes() {
        // Context/generation attention ops (names carry "attention").
        let ctx_attn = Op::ContextAttention(ContextAttentionOp::new(
            "context_attention",
            8,
            1,
            128,
            crate::common::enums::KvCacheQuantMode::Bfloat16,
            crate::common::enums::FmhaQuantMode::Bfloat16,
        ));
        assert_eq!(side_of(&ctx_attn), AfdSide::Attn);
        let gen_attn = Op::GenerationAttention(GenerationAttentionOp::new(
            "generation_attention",
            8,
            1,
            128,
            crate::common::enums::KvCacheQuantMode::Bfloat16,
        ));
        assert_eq!(side_of(&gen_attn), AfdSide::Attn);
        // MoE op (name carries "moe").
        let moe = Op::Moe(MoeOp::new(
            "moe",
            5120,
            768,
            8,
            128,
            1,
            8,
            crate::common::enums::MoeQuantMode::Bfloat16,
            "uniform",
        ));
        assert_eq!(side_of(&moe), AfdSide::Ffn);
        // A Mamba2 VARIANT rejects even with a neutral name (variant-first,
        // stronger than the Python name-only check).
        let mamba = Op::Mamba2(Mamba2Op {
            name: "mixer".into(),
            scale_factor: 1.0,
            kernel_source: "causal_conv1d_fn".into(),
            phase: "context".into(),
            d_model: 5120,
            d_state: 128,
            d_conv: 4,
            nheads: 64,
            head_dim: 64,
            n_groups: 1,
            chunk_size: 256,
        });
        assert!(classify_op(&mamba, false, AfdSide::Attn).is_err());
        // CustomAllReduce variant skips.
        let ar = Op::CustomAllReduce(CustomAllReduceOp::new("tp_ar", 1.0, 5120, 8));
        assert_eq!(side_of(&ar), AfdSide::Skip);
    }
}
