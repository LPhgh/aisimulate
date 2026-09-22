<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `nvfp4_wo` has no collected data: FP4 checkpoints fail validation on non-Blackwell systems in the default database mode

Status: **open, unassigned.** Needs a maintainer decision on whether
`nvfp4_wo` is meant to be a synthesized-only lane or a collected one; that
answer determines whether the short-term fix below is terminal or
transitional. No tracker ID yet.

## Symptom

An NVFP4 checkpoint on any non-Blackwell system is rejected by
`Task.validate()` in the default configuration:

```python
Task(serving_mode="agg", model_path="nvidia/DeepSeek-V3.1-NVFP4",
     system_name="h200_sxm", backend_name="trtllm", total_gpus=8).validate()
```

```
ValueError: Unsupported gemm quant mode 'nvfp4_wo' for system='h200_sxm',
backend='trtllm', version='1.3.0rc20'. Supported gemm modes:
['bfloat16', 'fp8', 'fp8_block', 'fp8_static']
```

The same task passes with `database_mode="HYBRID"` or
`database_mode="EMPIRICAL"`.

This is **not** an AFD or heterogeneous-pool problem. It reproduces on a
plain `serving_mode="agg"` task. It became visible while reviewing the
heterogeneous-AFD work because
`9f144c08` ("fix(afd): resolve the F pool's quant modes against its own
system") used `h200_sxm` attention + `b200_sxm` FFN on an NVFP4 checkpoint
as its headline example; that pairing validates and prices correctly under
HYBRID/EMPIRICAL, and fails under the default mode for the reason below.

## Root cause

Three facts combine into a guaranteed rejection.

1. **`nvfp4_wo` has zero collected rows anywhere.** Surveying every
   `gemm_perf.parquet` and MoE table under
   `python/aisimulate/src/aiconfigurator_core/systems/data/<system>/<backend>/<version>/`
   finds `nvfp4` on the Blackwell-family systems (`b200_sxm`, `b300_sxm`,
   `gb200`, `gb300`, `rtx_pro_6000_server`) and **no `nvfp4_wo` on any
   system, backend, or version** — gemm and MoE alike. It is a derived
   profile (`QuantMapping(memory=0.5625, compute=1)`) meant to be served by
   the transfer ladder, which is what the ladder's own comment says
   (`task_v2.py`, `_apply_fmha_data_fallback`): "The transfer ladder models
   the Marlin-class memory savings via the (0.5625, 1) util-level entry."

2. **The ladder produces it unconditionally.** `_apply_fmha_data_fallback`
   rewrites `nvfp4` → `nvfp4_wo` for gemm and MoE on every non-Blackwell
   system, with no reference to `database_mode` and no check that the
   target lane can actually be served.

3. **Validation only admits synthesized quants outside SILICON.**
   `_check_role_against_db` gates its XQUANT/XPROFILE relaxation on
   `_non_silicon = self.database_mode not in (None, DatabaseMode.SILICON.name)`,
   and the default `database_mode` is `None` — treated as SILICON.

So the ladder always emits a mode that only a non-SILICON mode can serve,
while the default mode is SILICON.

## Why the error message misleads

The ladder rewrites the mode before validation runs, so the rejection always
names `nvfp4_wo` — a mode that appears in no user-facing documentation of the
checkpoint. It names it even when the user pinned the mode explicitly:

```python
Task(serving_mode="agg", model_path="nvidia/DeepSeek-V3.1-NVFP4",
     system_name="h200_sxm", backend_name="trtllm", total_gpus=8,
     gemm_quant_mode=GEMMQuantMode.nvfp4).validate()
```

still resolves to `nvfp4_wo` and reports `Unsupported gemm quant mode
'nvfp4_wo'`. Rewriting an explicit value is deliberate here — the absence of
FP4 tensor cores is a hardware fact, not an inference the user can override
— so the defect is only that the resulting diagnostic points at the rewrite
instead of at the gap.

The actionable facts are that this hardware has no FP4 tensor cores, and
that the FP4 checkpoint can therefore only be modeled through the
weight-only transfer ladder, which needs a non-SILICON mode. Neither
appears. `_check` does already have a preflight that states the first one
plainly — `get_quant_tc_flops` raises `quant mode 'nvfp4' needs
'fp4_tc_flops', which this system's YAML does not define` — but the ladder
has consumed `nvfp4` by the time validation runs, so that branch is not
reachable here. Option (a) below makes it reachable again by not rewriting
in SILICON.

## Options

**(a) Make the ladder mode-aware.** Only rewrite `nvfp4` → `nvfp4_wo` when
the transfer ladder can serve it; in SILICON leave `nvfp4`, so validation
fails on the `fp4_tc_flops` preflight above, and extend that message to
name `--database-mode HYBRID` as the way to model the weight-only path.
Diagnostics only — no modeling change. Expected to leave goldens untouched,
since every config it affects already fails validation and therefore cannot
appear in a pinned parity case; that expectation needs a parity run to
confirm, not just the argument. Residual risk: a caller that skips
`validate()` would see `nvfp4` instead of `nvfp4_wo` in SILICON. Both fail,
differently.

**(b) Relax the `_non_silicon` gate** so SILICON can serve `nvfp4_wo`
through the util-level entry. Not recommended: it breaks SILICON's
"measured data only" contract and moves numbers, which puts goldens and the
parity suites in scope.

**(c) Collect `nvfp4_wo` rows on Hopper.** The only way SILICON can
genuinely support it. This is a Collector change and carries the gates in
`.claude/rules/collector/`; it may also be redundant with the transfer
model. Requires the maintainer decision named at the top.

**(d) Default `database_mode` to HYBRID for FP4-on-non-Blackwell.**
Rejected: silently changes default semantics.

Recommendation: **(a)** now, and put **(c)** to the maintainers as the
question "is `nvfp4_wo` a collected lane or a synthesized-only one?"

## Test gap

Nothing pins the SILICON + FP4-checkpoint + non-Blackwell path, which is why
a commit message could name a pairing that does not run under default
settings without anything failing. Whichever option is chosen should add a
test asserting the outcome for that combination — either a clear,
actionable rejection (a), or a successful resolution (c).

## Reproducing the survey

```python
import glob
import pandas as pd

D = "python/aisimulate/src/aiconfigurator_core/systems/data"
for path in sorted(glob.glob(D + "/*/gemm/*/*/gemm_perf.parquet")):
    parts = path.split("/")
    dtypes = sorted(set(pd.read_parquet(path, columns=["gemm_dtype"])["gemm_dtype"]))
    print(parts[-5], parts[-3], parts[-2], dtypes)
```
