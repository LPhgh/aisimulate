// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AFD (Attention-FFN Disaggregation) perf-model surface.
//!
//! Native numeric layer (see the Terminal State ADR in
//! `docs/sweeper/afd-topology.md`):
//!
//! - `partition`: A/F op-list partition (Op-variant first,
//!   name-marker fallback);
//! - `crate::operators::afd_comm`: the four AFD comm ops as real `Op`
//!   variants (schema 22);
//! - `model`: `AfdForwardPassPerfModel`, the `"aic-afd"` timing
//!   provider — one `best_available` forward-pass model per pool system,
//!   per-pool quant re-resolution, cross-SKU fabric model.

pub mod model;
pub mod partition;
