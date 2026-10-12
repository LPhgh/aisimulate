// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `AfdTimingModel` — the `"aic-afd"` provider as a replay `TimingModel`.
//! Lives in the engine tree (the
//! `TimingModel` trait is engine-layer) and consumes
//! [`crate::perfmodel::afd::model::AfdForwardPassPerfModel`].
//!
//! The provider freezes the measurement identity at construction: it is
//! measured once at `(a_batch_size, isl, osl, prefix=0)` and every pass
//! charges the frozen evaluation — a partial batch still pays the
//! full-batch step, and every decode pass pays the stride-averaged step.
//! That frozen identity covers
//! fixed synthetic workloads. Trace and agentic workloads instead use the
//! instantaneous per-pass evaluation: the pipeline is priced at the current
//! mean KV position with the effective microbatch count clamped to what the
//! actual pool-wide in-flight batch can fill
//! ([`AfdForwardPassPerfModel::effective_microbatch_geometry`]). Prefill
//! pricing keeps the frozen batch geometry for every workload kind.

use std::sync::Mutex;

use anyhow::Result;

use crate::engine::scheduler::afd::pipeline::{AfdEvalPhase, AfdPipelineKnobs, evaluate_afd_phase};
#[cfg(feature = "python")]
use crate::engine::timing::{
    TimingEvidenceAccumulator, TimingEvidenceSource, TimingOperationEvidence, ValidatedTimingPhase,
};
use crate::engine::timing::{TimingEvidenceSummary, TimingModel};
use crate::perfmodel::afd::model::{AfdForwardPassPerfModel, AfdPhase};

/// Replay timing provider for AFD ranks.
pub struct AfdTimingModel {
    perf: AfdForwardPassPerfModel,
    knobs: AfdPipelineKnobs,
    isl: u32,
    osl: u32,
    /// Island-frozen decode identity (fixed synthetic workloads); trace and
    /// agentic workloads use the instantaneous per-pass evaluation.
    frozen_decode_identity: bool,
    decode_step_ms: Mutex<Option<f64>>,
    /// Power evidence accumulator: per-op energy from the two
    /// pool engines flows into the standard replay power statistics.
    #[cfg(feature = "python")]
    evidence: Mutex<TimingEvidenceAccumulator>,
    /// Evidence computation cache (per phase + workload point) — the op
    /// walks are measured once, the cached phase evidence records per call.
    #[cfg(feature = "python")]
    evidence_cache:
        Mutex<std::collections::HashMap<(bool, u32, u32), std::sync::Arc<ValidatedTimingPhase>>>,
}

impl AfdTimingModel {
    /// Full constructor with the decode-identity choice.
    pub fn with_decode_identity(
        perf: AfdForwardPassPerfModel,
        isl: u32,
        osl: u32,
        frozen: bool,
    ) -> Self {
        let config = perf.config();
        let n_a_workers =
            (config.a_pool.n_a_nodes * config.gpus_per_node / config.a_pool.tp_a.max(1)).max(1);
        let n_f_workers = config.f_pool.n_f_nodes * config.gpus_per_node;
        Self {
            knobs: AfdPipelineKnobs {
                num_microbatches: config.num_microbatches,
                pipeline_model: config.pipeline_model,
                comm_overhead_factor: config.comm_overhead_factor,
                total_batch_size: n_a_workers * config.a_pool.a_batch_size,
                total_gpus: n_a_workers * config.a_pool.tp_a + n_f_workers,
                latency_correction: 1.0,
            },
            perf,
            isl,
            osl,
            frozen_decode_identity: frozen,
            decode_step_ms: Mutex::new(None),
            #[cfg(feature = "python")]
            evidence: Mutex::new(TimingEvidenceAccumulator::default()),
            #[cfg(feature = "python")]
            evidence_cache: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Per-op energy/latency evidence for one phase: compute ops carry
    /// their compiled per-step values; the four comm ops scale from
    /// per-layer to per-pass by `num_layers` (work-view energy — pipeline
    /// overlap hides latency, not joules).
    #[cfg(feature = "python")]
    fn phase_evidence(
        &self,
        prefill: bool,
        isl: u32,
        prefix: u32,
    ) -> Result<std::sync::Arc<ValidatedTimingPhase>> {
        let key = (prefill, isl, prefix);
        if let Some(cached) = self
            .evidence_cache
            .lock()
            .expect("evidence cache poisoned")
            .get(&key)
        {
            return Ok(std::sync::Arc::clone(cached));
        }
        let num_layers = self.perf.config().num_layers.max(1) as f64;
        let phase = if prefill {
            AfdPhase::Prefill
        } else {
            AfdPhase::Decode
        };
        let mut operations = Vec::new();
        for op in self
            .perf
            .collect_phase_operations(phase, isl, self.osl, prefix)?
        {
            let scale = if op.per_layer { num_layers } else { 1.0 };
            operations.push(TimingOperationEvidence::new(
                op.name,
                op.latency_ms * scale,
                (op.energy_wms > 0.0).then_some(op.energy_wms * scale),
                TimingEvidenceSource::from_native(op.source.as_str()),
            )?);
        }
        let evidence = std::sync::Arc::new(ValidatedTimingPhase::from_operations(operations)?);
        self.evidence_cache
            .lock()
            .expect("evidence cache poisoned")
            .insert(key, std::sync::Arc::clone(&evidence));
        Ok(evidence)
    }

    #[cfg(feature = "python")]
    fn record_phase_evidence(&self, prefill: bool, isl: u32, prefix: u32) -> Result<()> {
        let phase = self.phase_evidence(prefill, isl, prefix)?;
        self.evidence
            .lock()
            .map_err(|_| anyhow::anyhow!("AFD timing evidence accumulator was poisoned"))?
            .record(&phase, prefill)
    }
}

impl TimingModel for AfdTimingModel {
    fn predict_prefill_ms(
        &self,
        _batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        let metrics = self.perf.predict_phase_metrics(
            AfdPhase::Prefill,
            mean_isl as u32,
            self.osl,
            mean_prefix as u32,
        )?;
        let evaluation = evaluate_afd_phase(
            &self.knobs,
            &metrics.layer_times,
            AfdEvalPhase::Prefill,
            mean_isl as u32,
            self.osl,
        )?;
        #[cfg(feature = "python")]
        self.record_phase_evidence(true, mean_isl as u32, mean_prefix as u32)?;
        Ok(evaluation.step_latency_ms)
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        _active_kv_tokens: usize,
        mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<f64> {
        if !self.frozen_decode_identity {
            // Trace/agentic: instantaneous pipeline evaluation at the current
            // mean KV position (per-pass semantics; the frozen stride average
            // only makes sense for the fixed synthetic identity). The
            // effective microbatch count is clamped to what the actual
            // pool-wide batch can fill (continuous-batching semantics; the
            // frozen identity keeps charging the full-depth pipeline).
            let metrics = self.perf.predict_decode_step_ms(
                batch_size as u32,
                mean_context_length.max(1) as u32,
                0,
            )?;
            let (m_eff, _) = self.perf.effective_microbatch_geometry(batch_size as u32);
            let knobs = AfdPipelineKnobs {
                num_microbatches: m_eff,
                ..self.knobs
            };
            let evaluation = evaluate_afd_phase(
                &knobs,
                &metrics.layer_times,
                AfdEvalPhase::Decode,
                self.isl,
                self.osl,
            )?;
            #[cfg(feature = "python")]
            // Evidence belongs to the same workload point the prediction
            // evaluated (the current mean KV position), not the frozen
            // identity: trace traffic's placeholder self.isl would misreport
            // the decode energy.
            self.record_phase_evidence(false, mean_context_length.max(1) as u32, 0)?;
            return Ok(evaluation.step_latency_ms);
        }
        if let Some(step) = *self.decode_step_ms.lock().expect("decode cache poisoned") {
            // The op walk is cached, but the evidence record mirrors the
            // monolithic providers: every predicted pass accumulates.
            #[cfg(feature = "python")]
            self.record_phase_evidence(false, self.isl, 0)?;
            return Ok(step);
        }
        let metrics = self
            .perf
            .predict_phase_metrics(AfdPhase::Decode, self.isl, self.osl, 0)?;
        let evaluation = evaluate_afd_phase(
            &self.knobs,
            &metrics.layer_times,
            AfdEvalPhase::Decode,
            self.isl,
            self.osl,
        )?;
        let step = evaluation.step_latency_ms;
        *self.decode_step_ms.lock().expect("decode cache poisoned") = Some(step);
        #[cfg(feature = "python")]
        self.record_phase_evidence(false, self.isl, 0)?;
        Ok(step)
    }

    fn evidence_summary(&self) -> Option<TimingEvidenceSummary> {
        #[cfg(feature = "python")]
        {
            return self
                .evidence
                .lock()
                .ok()
                .map(|evidence| evidence.snapshot());
        }
        #[cfg(not(feature = "python"))]
        None
    }

    #[cfg(feature = "python")]
    fn reset_evidence(&self) -> Result<()> {
        *self
            .evidence
            .lock()
            .map_err(|_| anyhow::anyhow!("AFD timing evidence accumulator was poisoned"))? =
            TimingEvidenceAccumulator::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::CommQuantMode;
    use crate::perf_database::PerfDatabase;
    use crate::perfmodel::afd::model::{
        AfdAPoolSpec, AfdEstimatorPlumbing, AfdFPoolSpec, AfdForwardPassModelConfig,
        AfdForwardPassPerfModel, AfdPipelineModel, AfdQuantOverrides,
    };
    use crate::perfmodel::engine::Engine;
    use crate::perfmodel::engine::spec::EngineSpec;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aisimulate_core/systems")
    }

    fn golden_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parity_tests/perfmodel/goldens")
    }

    fn build_engine(spec_file: &str, db: &Arc<PerfDatabase>) -> Arc<Engine> {
        let path = golden_dir().join("afd_specs").join(spec_file);
        let json = std::fs::read_to_string(&path).expect("read spec fixture");
        let spec: EngineSpec = serde_json::from_str(&json).expect("parse spec");
        Arc::new(Engine::build(spec, db.clone()).expect("engine build"))
    }

    /// The golden-case provider (pure h200 pool pair, frozen identity at
    /// isl=1024/osl=128) shared by the parity and evidence tests.
    fn provider(frozen: bool) -> AfdTimingModel {
        provider_for(1, 18, 3, frozen)
    }

    fn provider_for(
        n_a_nodes: u32,
        a_batch_size: u32,
        num_microbatches: u32,
        frozen: bool,
    ) -> AfdTimingModel {
        let db = Arc::new(
            PerfDatabase::load(&systems_root(), "h200_sxm", "trtllm", "1.3.0rc20")
                .expect("load db"),
        );
        let a_engine = build_engine("pure_h200_a.json", &db);
        let f_engine = build_engine("pure_h200_f.json", &db);
        let config = AfdForwardPassModelConfig {
            model: "Qwen/Qwen3-32B".into(),
            backend: crate::BackendKind::Trtllm,
            backend_version: Some("1.3.0rc20".into()),
            a_pool: AfdAPoolSpec {
                system: "h200_sxm".into(),
                tp_a: 8,
                n_a_nodes,
                a_batch_size,
            },
            f_pool: AfdFPoolSpec {
                system: "h200_sxm".into(),
                f_moe_ep_size: 1,
                n_f_nodes: 1,
            },
            gpus_per_node: 8,
            num_microbatches,
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
        };
        let perf = AfdForwardPassPerfModel::from_engines(config, a_engine, f_engine)
            .expect("afd perf model");
        AfdTimingModel::with_decode_identity(perf, 1024, 128, frozen)
    }

    fn frozen_provider() -> AfdTimingModel {
        provider(true)
    }

    /// Continuous-batching clamp: the instantaneous decode step shrinks as
    /// the pool-wide in-flight batch contracts (fewer microbatches fill the
    /// pipeline), while the frozen identity stays batch-independent.
    #[test]
    fn instantaneous_decode_clamps_microbatches_as_batch_shrinks() {
        let timing = provider(false);
        let step = |batch: usize| {
            timing
                .predict_decode_ms(batch, batch * 1100, 1100, 1 << 20)
                .expect("decode")
        };
        let full = step(18);
        let two_micro = step(12);
        let one_micro = step(6);
        let single = step(1);
        assert!(
            full > two_micro && two_micro > one_micro && one_micro > single,
            "step must shrink with the in-flight batch: full={full} two={two_micro} one={one_micro} single={single}"
        );

        // A full batch keeps the configured pipeline depth: identical to the
        // unclamped knobs evaluation at the same KV position.
        let metrics = timing
            .perf
            .predict_decode_step_ms(18, 1100, 0)
            .expect("metrics");
        let unclamped = evaluate_afd_phase(
            &timing.knobs,
            &metrics.layer_times,
            AfdEvalPhase::Decode,
            1024,
            128,
        )
        .expect("eval");
        assert!((full - unclamped.step_latency_ms).abs() < 1e-12 * full);

        let frozen = frozen_provider();
        let frozen_full = frozen
            .predict_decode_ms(18, 18 * 1024, 1024, 1 << 20)
            .expect("frozen full");
        let frozen_partial = frozen
            .predict_decode_ms(6, 6 * 1024, 1024, 1 << 20)
            .expect("frozen partial");
        assert_eq!(frozen_full, frozen_partial);
    }

    /// Pool-wide batch contract: with two A workers the provider splits the
    /// pool-wide batch per worker, so identical per-worker micro geometry
    /// prices identically. (Before the contract fix the instantaneous path
    /// treated the pool-wide count as the per-worker micro batch, inflating
    /// both sides quadratically in n_a_workers.) The dense FFN is
    /// weight-bound at these batch sizes, so the assertions key on equal
    /// geometry producing equal price rather than on latency slopes.
    #[test]
    fn instantaneous_decode_splits_pool_batch_across_a_workers() {
        let single = provider_for(1, 18, 3, false);
        let dual = provider_for(2, 18, 3, false);
        assert_eq!(single.perf.effective_microbatch_geometry(18), (3, 6));
        assert_eq!(dual.perf.effective_microbatch_geometry(36), (3, 6));
        assert_eq!(dual.perf.effective_microbatch_geometry(18), (2, 5));

        let one = single
            .perf
            .predict_decode_step_ms(18, 1100, 0)
            .expect("single-worker metrics");
        let two = dual
            .perf
            .predict_decode_step_ms(36, 1100, 0)
            .expect("dual-worker metrics");
        assert!(
            (one.layer_times.attention_ms - two.layer_times.attention_ms).abs() < 1e-12,
            "same per-worker micro geometry must price the A side identically: {} vs {}",
            one.layer_times.attention_ms,
            two.layer_times.attention_ms
        );

        // The dual pool's F side serves b = n_a_workers * a_micro = 12: the
        // same total the single-worker pool reaches only at a_micro = 12.
        let single_b12 = single
            .perf
            .predict_decode_step_ms(36, 1100, 0)
            .expect("single-worker b=12 metrics");
        assert!(
            (two.layer_times.ffn_ms - single_b12.layer_times.ffn_ms).abs() < 1e-12,
            "dual-pool F side must run at b=12 like the single pool at a_micro=12: {} vs {}",
            two.layer_times.ffn_ms,
            single_b12.layer_times.ffn_ms
        );
        assert_eq!(single.perf.effective_microbatch_geometry(36), (3, 12));
    }

    /// End-to-end provider oracle: the AFD perf model over the compiled
    /// per-pool specs drives the timing provider to the exact frozen golden
    /// step latencies (golden e2e means:
    /// mean_ttft_ms=422.1866311464918, mean_tpot_ms=11.846695674852583).
    #[test]
    fn provider_reproduces_island_frozen_step_latencies() {
        let timing = frozen_provider();

        // Prefill: the provider freezes the measurement; ANY scheduler
        // geometry returns the same frozen step.
        let prefill = timing.predict_prefill_ms(8, 1024, 0).expect("prefill");
        assert!((prefill - 422.1866311464918).abs() < 1e-9 * 422.2);
        let prefill_partial = timing
            .predict_prefill_ms(3, 1024, 0)
            .expect("prefill partial");
        assert_eq!(prefill, prefill_partial);

        // Decode: the stride-averaged frozen step, constant per pass.
        let decode = timing
            .predict_decode_ms(8, 8 * 1024, 1024, 16384)
            .expect("decode");
        assert!((decode - 11.846695674852583).abs() < 1e-9 * 11.85);
        let decode_later = timing
            .predict_decode_ms(8, 8 * 1100, 1100, 16384)
            .expect("decode later");
        assert_eq!(decode, decode_later);
    }

    /// The frozen decode-step cache must not swallow evidence: like the
    /// monolithic `AicTimingModel`, every predicted pass accumulates into
    /// the summary (power statistics are per-replay totals).
    #[cfg(feature = "python")]
    #[test]
    fn frozen_decode_records_evidence_per_call() {
        let timing = frozen_provider();
        timing.reset_evidence().expect("reset");
        timing
            .predict_decode_ms(8, 8 * 1024, 1024, 16384)
            .expect("decode");
        let one_pass = timing
            .evidence_summary()
            .expect("summary")
            .decode
            .latency_ms;
        assert!(one_pass > 0.0, "one decode pass must record evidence");
        for _ in 0..4 {
            timing
                .predict_decode_ms(8, 8 * 1024, 1024, 16384)
                .expect("decode");
        }
        let total = timing
            .evidence_summary()
            .expect("summary")
            .decode
            .latency_ms;
        assert!(
            (total - 5.0 * one_pass).abs() < 1e-9 * one_pass,
            "five decode calls must accumulate five passes: total={total} one={one_pass}"
        );
    }

    /// Trace/agentic (instantaneous) decode: each pass's evidence must be
    /// recorded at the pass's own prediction point (the current mean KV
    /// position), not the frozen identity — the placeholder self.isl of
    /// variable-length traffic would misreport the decode energy.
    #[cfg(feature = "python")]
    #[test]
    fn instantaneous_decode_records_evidence_at_the_prediction_point() {
        let timing = provider(false);
        timing.reset_evidence().expect("reset");
        timing
            .predict_decode_ms(8, 8 * 1100, 1100, 16384)
            .expect("decode at 1100");
        let first_total = timing
            .evidence_summary()
            .expect("summary")
            .decode
            .latency_ms;
        timing
            .predict_decode_ms(8, 8 * 2000, 2000, 16384)
            .expect("decode at 2000");
        let second_total = timing
            .evidence_summary()
            .expect("summary")
            .decode
            .latency_ms;

        let first_phase = timing
            .phase_evidence(false, 1100, 0)
            .expect("phase at 1100");
        let second_phase = timing
            .phase_evidence(false, 2000, 0)
            .expect("phase at 2000");
        let identity_phase = timing
            .phase_evidence(false, 1024, 0)
            .expect("identity phase");
        // The test is only meaningful if the point actually moves the phase.
        assert!(first_phase.as_phase().latency_ms != identity_phase.as_phase().latency_ms);
        let first_latency = first_phase.as_phase().latency_ms;
        assert!(
            (first_total - first_latency).abs() < 1e-9 * first_latency,
            "pass at 1100 recorded {first_total}ms, phase at 1100 is {first_latency}ms"
        );
        let increment = second_total - first_total;
        let second_latency = second_phase.as_phase().latency_ms;
        assert!(
            (increment - second_latency).abs() < 1e-9 * second_latency,
            "pass at 2000 recorded {increment}ms, phase at 2000 is {second_latency}ms"
        );
    }
}
