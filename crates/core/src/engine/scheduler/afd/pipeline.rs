// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AFD pipeline evaluation.
//!
//! The per-phase AFD pipeline cadence the scheduler core consumes at pass
//! planning time. Notes:
//!
//! - `comm_overhead_factor` multiplies the two transfer terms HERE, exactly
//!   once; the measurement side (`AfdForwardPassPerfModel`) keeps 1.0.
//! - The serial/conservative/optimistic cycle/fill/step formulas are
//!   carried over term by term, including the `num_microbatches == 1`
//!   serial rule (mirrored on the measurement side by
//!   `AfdForwardPassPerfModel::pipeline_tcycle`) and the optimistic ->
//!   conservative degradation with its effective-model tracking.

use crate::common::error::AicError;
use crate::perfmodel::afd::model::{AfdLayerTimes, AfdPipelineModel};

/// Pipeline evaluation inputs beyond the layer times (the topology knobs the
/// formula reads).
#[derive(Clone, Copy, Debug)]
pub struct AfdPipelineKnobs {
    pub num_microbatches: u32,
    pub pipeline_model: AfdPipelineModel,
    pub comm_overhead_factor: f64,
    /// In-flight sequences across all A workers
    /// (`n_a_workers * a_batch_size`).
    pub total_batch_size: u32,
    /// Total GPUs (attention + FFN).
    pub total_gpus: u32,
    /// Multiplies the final step latency.
    pub latency_correction: f64,
}

impl Default for AfdPipelineKnobs {
    fn default() -> Self {
        Self {
            num_microbatches: 1,
            pipeline_model: AfdPipelineModel::NoOverlap,
            comm_overhead_factor: 1.0,
            total_batch_size: 1,
            total_gpus: 1,
            latency_correction: 1.0,
        }
    }
}

/// One phase's AFD pipeline evaluation with its effective scheduling model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AfdPhaseEvaluation {
    pub step_latency_ms: f64,
    pub sequence_rate: f64,
    pub tokens_per_second: f64,
    pub communication_hidden: bool,
    pub balance_ratio: f64,
    pub cycle_ms: f64,
    pub pipeline_fill_ms: f64,
    pub requested_pipeline_model: AfdPipelineModel,
    pub effective_pipeline_model: AfdPipelineModel,
    pub total_gpus: u32,
}

/// Whether the evaluated phase is decode (drives the
/// tokens-per-second/sequence-rate convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfdEvalPhase {
    Prefill,
    Decode,
}

/// Evaluate one AFD phase (serial cadence for one microbatch).
pub fn evaluate_afd_phase(
    knobs: &AfdPipelineKnobs,
    times: &AfdLayerTimes,
    phase: AfdEvalPhase,
    input_length: u32,
    output_length: u32,
) -> Result<AfdPhaseEvaluation, AicError> {
    if input_length < 1 || output_length < 1 {
        return Err(AicError::InvalidEngineConfig(format!(
            "input_length and output_length must be positive integers, got {input_length}/{output_length}"
        )));
    }
    if !knobs.latency_correction.is_finite() || knobs.latency_correction <= 0.0 {
        return Err(AicError::InvalidEngineConfig(format!(
            "latency_correction must be a positive finite number, got {}",
            knobs.latency_correction
        )));
    }

    let t_a = times.attention_ms;
    let t_f = times.ffn_ms;
    // The factor lands here, once.
    let t_a2f = times.a_to_f_ms * knobs.comm_overhead_factor;
    let t_f2a = times.f_to_a_ms * knobs.comm_overhead_factor;
    let t_c = t_a2f + t_f2a;
    let requested = knobs.pipeline_model;
    let num_microbatches = knobs.num_microbatches.max(1);

    let (cycle, effective, hidden) =
        if requested == AfdPipelineModel::NoOverlap || num_microbatches == 1 {
            // A single microbatch must return from F before its next layer can start.
            (t_a + t_a2f + t_f + t_f2a, AfdPipelineModel::NoOverlap, false)
        } else if requested == AfdPipelineModel::PartialOverlap {
            (
                (t_a + t_a2f).max(t_f + t_f2a),
                AfdPipelineModel::PartialOverlap,
                false,
            )
        } else {
            let min_microbatches = 2.0 + t_c / t_a.max(t_f).max(1e-9);
            if (num_microbatches as f64) < min_microbatches {
                (
                    (t_a + t_a2f).max(t_f + t_f2a),
                    AfdPipelineModel::PartialOverlap,
                    false,
                )
            } else {
                (
                    t_a.max(t_f).max(t_c),
                    AfdPipelineModel::FullOverlap,
                    t_c <= t_a.max(t_f),
                )
            }
        };

    let fill = t_a + t_f + t_a2f + t_f2a;
    let step = (fill + cycle * (num_microbatches * times.num_layers).saturating_sub(1) as f64)
        * knobs.latency_correction;
    if step <= 0.0 {
        return Err(AicError::InvalidEngineConfig(
            "AFD pipeline evaluation produced a non-positive step latency".into(),
        ));
    }
    let (sequence_rate, tokens_per_second) = match phase {
        AfdEvalPhase::Decode => {
            let tps = f64::from(knobs.total_batch_size) / (step / 1000.0);
            (tps / f64::from(output_length), tps)
        }
        AfdEvalPhase::Prefill => {
            let rate = f64::from(knobs.total_batch_size) / (step / 1000.0);
            (rate, rate * f64::from(input_length))
        }
    };
    Ok(AfdPhaseEvaluation {
        step_latency_ms: step,
        sequence_rate,
        tokens_per_second,
        communication_hidden: hidden,
        balance_ratio: t_a.min(t_f) / t_a.max(t_f).max(1e-9),
        cycle_ms: cycle,
        pipeline_fill_ms: fill,
        requested_pipeline_model: requested,
        effective_pipeline_model: effective,
        total_gpus: knobs.total_gpus,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Oracle: hand-computed from the frozen golden layer times
    /// (`goldens/afd_layer_times.json`, qwen3-32b-h200-trtllm-pure-both)
    /// through the published formulas, cross-checked against the golden
    /// end-to-end means (mean_tpot_ms=11.846695674852583,
    /// mean_ttft_ms=422.1866311464918 in `goldens/afd_e2e.json`).
    fn golden_decode_times() -> AfdLayerTimes {
        AfdLayerTimes {
            attention_ms: 0.06126769722822783,
            ffn_ms: 0.06084020703299543,
            a_to_f_ms: 0.0112288,
            f_to_a_ms: 0.0112288,
            num_layers: 64,
        }
    }

    fn golden_prefill_times() -> AfdLayerTimes {
        AfdLayerTimes {
            attention_ms: 0.46361824477271635,
            ffn_ms: 0.9215771421964521,
            a_to_f_ms: 1.2682912,
            f_to_a_ms: 1.2682912,
            num_layers: 64,
        }
    }

    fn pure_knobs() -> AfdPipelineKnobs {
        AfdPipelineKnobs {
            num_microbatches: 3,
            pipeline_model: AfdPipelineModel::FullOverlap,
            comm_overhead_factor: 1.0,
            total_batch_size: 18,
            total_gpus: 16,
            latency_correction: 1.0,
        }
    }

    #[test]
    fn decode_optimistic_holds_when_microbatches_fill_pipeline() {
        let eval = evaluate_afd_phase(
            &pure_knobs(),
            &golden_decode_times(),
            AfdEvalPhase::Decode,
            1024,
            128,
        )
        .expect("decode eval");
        // min_m = 2 + 0.0224576 / 0.0612677 = 2.3665 < 3 -> optimistic holds;
        // cycle = t_a; step = fill + t_a * 191.
        assert_eq!(eval.effective_pipeline_model, AfdPipelineModel::FullOverlap);
        assert!(eval.communication_hidden);
        assert!((eval.step_latency_ms - 11.846695674852583).abs() < 1e-9 * 11.85);
        assert!((eval.balance_ratio - 0.9930225842561056).abs() < 1e-12);
        // decode: tokens_per_second = 18 / (step/1000); rate = tps / 128.
        let tps = 18.0 / (eval.step_latency_ms / 1000.0);
        assert!((eval.tokens_per_second - tps).abs() < 1e-12 * tps);
        assert!((eval.sequence_rate - tps / 128.0).abs() < 1e-12);
    }

    #[test]
    fn prefill_optimistic_degrades_to_conservative() {
        let eval = evaluate_afd_phase(
            &pure_knobs(),
            &golden_prefill_times(),
            AfdEvalPhase::Prefill,
            1024,
            128,
        )
        .expect("prefill eval");
        // min_m = 2 + 2.5365824 / 0.9215771 = 4.7525 > 3 -> conservative;
        // cycle = max(t_a + a2f, t_f + f2a) = t_f + f2a; step = fill + cycle * 191.
        assert_eq!(
            eval.effective_pipeline_model,
            AfdPipelineModel::PartialOverlap
        );
        assert!(!eval.communication_hidden);
        assert!((eval.step_latency_ms - 422.1866311464918).abs() < 1e-9 * 422.2);
        // prefill: rate = 18 / (step/1000); tps = rate * 1024.
        let rate = 18.0 / (eval.step_latency_ms / 1000.0);
        assert!((eval.sequence_rate - rate).abs() < 1e-12 * rate);
        assert!((eval.tokens_per_second - rate * 1024.0).abs() < 1e-9 * rate * 1024.0);
    }

    #[test]
    fn serial_and_single_microbatch_use_serial_cadence() {
        let knobs = AfdPipelineKnobs {
            pipeline_model: AfdPipelineModel::NoOverlap,
            ..pure_knobs()
        };
        let eval = evaluate_afd_phase(
            &knobs,
            &golden_decode_times(),
            AfdEvalPhase::Decode,
            1024,
            128,
        )
        .expect("serial eval");
        let cycle = 0.06126769722822783 + 0.0112288 + 0.06084020703299543 + 0.0112288;
        let step = 0.14456550426122326 + cycle * 191.0;
        assert!((eval.step_latency_ms - step).abs() < 1e-12 * step);
        assert_eq!(eval.effective_pipeline_model, AfdPipelineModel::NoOverlap);

        // num_microbatches == 1 forces serial even under optimistic request.
        let one_ub = AfdPipelineKnobs {
            num_microbatches: 1,
            ..pure_knobs()
        };
        let one_eval = evaluate_afd_phase(
            &one_ub,
            &golden_decode_times(),
            AfdEvalPhase::Decode,
            1024,
            128,
        )
        .expect("one-ub eval");
        assert_eq!(one_eval.effective_pipeline_model, AfdPipelineModel::NoOverlap);
        let one_step = 0.14456550426122326 + cycle * 63.0;
        assert!((one_eval.step_latency_ms - one_step).abs() < 1e-12 * one_step);
    }

    #[test]
    fn comm_overhead_factor_lands_once_on_transfers() {
        let scaled = AfdPipelineKnobs {
            comm_overhead_factor: 1.5,
            ..pure_knobs()
        };
        let eval = evaluate_afd_phase(
            &scaled,
            &golden_decode_times(),
            AfdEvalPhase::Decode,
            1024,
            128,
        )
        .expect("scaled eval");
        let t_c = 0.0112288 * 1.5 * 2.0;
        let cycle = 0.06126769722822783_f64.max(0.06084020703299543).max(t_c);
        let fill = 0.06126769722822783 + 0.06084020703299543 + t_c;
        let step = fill + cycle * 191.0;
        assert!((eval.step_latency_ms - step).abs() < 1e-12 * step);
    }

    #[test]
    fn invalid_inputs_reject() {
        assert!(
            evaluate_afd_phase(
                &pure_knobs(),
                &golden_decode_times(),
                AfdEvalPhase::Decode,
                0,
                128
            )
            .is_err()
        );
        let bad_correction = AfdPipelineKnobs {
            latency_correction: 0.0,
            ..pure_knobs()
        };
        assert!(
            evaluate_afd_phase(
                &bad_correction,
                &golden_decode_times(),
                AfdEvalPhase::Decode,
                1024,
                128
            )
            .is_err()
        );
    }
}
