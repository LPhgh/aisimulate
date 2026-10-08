# AFD estimate regression

This CPU-only suite runs fixed queries through the real
`aisimulate.legacy_cli.api.cli_estimate` entry point and compares public latency
outputs to frozen targets. It uses the native runtime and bundled performance
database without mocking estimator results.

From the repository root:

```bash
uv sync --project python/aisimulate --extra dev
python/aisimulate/.venv/bin/python -m pytest -p no:timeout \
  tests/e2e/test_afd_estimate_regression.py \
  --basetemp=/tmp/afd-estimate-regression --junitxml=/tmp/afd-estimate-results.xml
```

Use a dedicated `--basetemp` directory: pytest clears it on each run. Each
executed query writes a JSON record there with the resolved query, baseline
commit, input commit, target, actual, signed difference, relative difference, path outputs,
and captured logs. Pytest reports preflight failures separately.

## Configuration and coverage

`afd_estimate_baseline.json` pins Qwen/Qwen3-235B-A22B-FP8 on H200 with SGLang
0.5.14 in HYBRID mode. The common configuration uses one eight-GPU attention
node, one eight-GPU FFN node, attention TP=1, FFN EP=8, three microbatches,
ISL=16000, OSL=64 and six requests per attention worker.

The suite contains 34 query cases (21 numerical and 13 rejection cases), one
manifest-integrity test, and one strict-xfail test for a known AFD+MTP defect:

- Conservative, optimistic and serial decode.
- Uneven microbatches and optimistic fallback with two microbatches.
- Prefill and both phases.
- Microbatch counts 1 and 4, attention TP 2 and 4, FFN EP 4, and two-node A/F pools.
- Short/long context, communication factors 0.5/1, FFN boundary assignment, and PD combinations.
- Invalid pipeline/phase, incompatible both-phase/PD, missing A/F node counts,
  zero microbatches, and zero attention TP.
- Non-divisor FFN EP and attention TP, zero attention nodes, zero `a_batch_size`,
  zero communication factor, and an out-of-memory configuration.

Standalone decode compares TPOT, prefill compares TTFT, and both checks each
separately. PD combinations check both metrics, complementary static/AFD paths,
and the 24-GPU total budget.
Targets must be positive and finite. Public SDK latencies are rounded to three
decimal milliseconds, and the tests compare that output using `actual == target`.
They also check phase coverage, pipeline metadata, communication hiding,
microbatch size, worker topology, and rejection type/message/origin.

The integrity test rejects missing, duplicate or unexpected case IDs and
missing required metrics, path or phase metadata. Individual numerical cases
also reject an empty or phase-inappropriate metrics dictionary. Model
configuration, system YAML and H200 database files are checked against SHA-256
fingerprints, including the database file inventory.

## Baseline lifecycle and scope

The top-level `commit` produced the original nine targets; each added case
records its own `target_commit` (`69165795` for the 19 added cases).
Independent subsequent runs verify deterministic regression behavior. Tests never
regenerate targets.
Changes to model/database inputs or targets require explicit review and a new
baseline capture. The current commit field records baseline provenance; it does
not select or check out that revision during execution.

This is a 34-query matrix for the current API. It is not a claim of one-to-one
coverage of an external design matrix. It does not implement measured per-step
accuracy comparison or attribution analysis. Passing these tests
does not establish agreement with measured GPU latency.

The repository Full CI contracts shard scans root `tests/`, so this suite is
eligible for collection once committed. Dedicated persistence of the per-case
JSON files is not configured here. The baseline records the capture Python
version but does not enforce a complete dependency or runtime fingerprint.

## Baseline provenance

The suite imports the renamed `aisimulate_core` package and the retained
`aisimulate.legacy_cli.api.cli_estimate` entry point.

`commit` records the original numerical-target provenance (captured with the
pre-rename `aiconfigurator` packages); `input_commit` records the revision
whose inputs were fingerprinted (`cab77b9c7888b1c40df8cd3228f5e03c355ef23c`).
The input refresh changed no numerical targets, path expectations, error
expectations or query settings. The H200 inventory gained 36 files and lost
two FPM files; seven retained inputs changed (system YAML and GEMM/MLA data
or metadata). All 204 current input files are fingerprinted.

The nine original targets were re-verified unchanged on the refreshed inputs
(Python 3.13.7) before the matrix was extended to 28 cases.

## Commit-to-commit comparison

`scripts/compare_afd_commits.py` executes this SAME manifest in two independently
prepared checkouts. It supports both the pre-rename and current Python packages.
Prepare each checkout with `uv sync --project python/aisimulate --extra dev`.
Runtime source/configuration must be clean; the script verifies the imported
Python module paths. It neither switches branches nor installs environments.

```bash
python scripts/compare_afd_commits.py \
  --before /absolute/path/to/before-checkout \
  --after /absolute/path/to/after-checkout \
  --systems-path /absolute/path/to/shared/systems \
  --output /absolute/path/to/new-comparison-directory
```

`--systems-path` is the directory containing `h200_sxm.yaml` and `data/`.
Use a version-controlled, unchanged directory for both runs. Without this option,
each checkout uses its bundled systems.

Two separate input gates apply, and they answer different questions:

- **Side-to-side equality.** The two snapshots must fingerprint the same inputs,
  otherwise the comparison mixes a code change with a data change.
- **Manifest baseline equality.** Each side must also match `input_sha256` in the
  manifest. Without this second gate, two checkouts could agree on a stale, partial
  or unrelated H200 database and the run would look like a clean comparison of code
  alone. Missing files are reported, not just changed hashes: a snapshot rebuilt by
  walking the systems directory simply omits absent files, so both sides can lose the
  same file.

The run is labelled `paired-code` only when **both** gates hold; otherwise it is
`code-and-data`. Agreeing with each other is not enough — the label must not advertise
an isolation the run does not have.

`--allow-input-drift` downgrades both gates to recorded findings for a deliberate
combined code-and-data comparison, which must not be interpreted as isolated code
impact. The differing keys are then listed in `report.md` and kept in full in
`comparison.json` under `input_drift`, so a tolerated drift is still auditable.

Separate Python executables can be selected with `--before-python` and
`--after-python`; otherwise each checkout's project venv is used. Interpreter
versions and installed package versions are recorded and now diffed into
`environment_diff` / a `report.md` section, but dependency equality is still not
enforced: attribution to code alone requires reviewing that diff.

The paired gate checks all 34 resolved queries, required metrics, positive finite
latencies, phase/path metadata and rejection type/message pattern/origin on BOTH
sides. By default it compares the two executions **directly and has no absolute
anchor**: comparing a commit against itself passes trivially, and a drift shared by
both sides is invisible. `--check-targets` adds that anchor by also gating each side
against the manifest's frozen targets. Regardless of the flag, each side's deviation
from target is computed into `target_deviations` and shown as two extra `report.md`
columns, and the report header states which anchor applied. Fixed-target
verification as a hard gate remains the separate pytest suite above.

The default relative tolerance is zero; any numerical change fails, including
faster results. `--relative-tolerance` is a fractional absolute relative change
bound (e.g. `0.01` = 1%) chosen before a run and recorded in its report. It is
single-sided relative to the before value, so `0.01` admits `+1%` but also `-0.99%`.

Every output directory must be new. Outputs are `before.json`, `after.json`, both
execution logs, `comparison.json`, and `report.md`. The report carries the
per-metric before/after, signed millisecond delta, percentage delta and each side's
deviation from the frozen target, plus an input-fingerprint drift section (only when
drift was recorded) and an environment-difference section. Commit IDs, input hashes,
query coverage and harness/manifest hashes are retained. Execution/setup failures
write `failure.json` and return nonzero; completed comparisons return 0 for pass or
2 for gate failure. Failures are never dropped from the denominator.

The comparator has 30 guard tests in `tests/test_afd_commit_comparison.py`; these
are harness tests, not extra workload cases. They cover the two input gates, the
`paired-code` labelling rule, the optional target anchor, environment-diff reporting,
report rendering, and the rule that a structural manifest/snapshot mismatch returns a
failure rather than escaping the strict zip as an exception. Pair execution is an
explicit command, not an automatic CI job comparing arbitrary commits.

`tests/test_afd_projection_contract.py` adds three native-runtime-free canary tests
for the AFD+MTP defect described under "Known uncovered scope" below.

### Verified run (2026-10-01)

Compared `36b5465faf9aa383223533580006f1e387225c45` against
`6916579586561f22fbde2f70a7f3d395c2ec5a96` using the latter checkout's
H200 systems directory for both executions: 28/28 cases passed, all 24 numerical
comparisons had zero delta, and all seven expected rejections matched.
Python was 3.13.7 on both sides. Environment differences were AISimulate
0.12.0/0.13.0, googleapis-common-protos 1.75.1/1.75.3, and ijson absent/3.5.1;
this establishes observed agreement for these environments, not isolated causal
attribution to source changes. Using each checkout's own bundled database was
correctly rejected by the default input-equality gate (exit 2).

That run predates both the matrix growth to 34 cases and the guard-test additions,
so its counts (28/28 cases, seven rejections, 44/44 tests, 15 comparator guard
tests) are left as observed rather than restated. Current counts, re-run locally
after rebuilding the native runtime offline: `test_afd_estimate_regression.py`
reports **35 passed + 1 xfailed** (34 workloads, one integrity check, one
strict-xfail AFD+MTP test), `test_afd_commit_comparison.py` reports **30 passed**,
and `test_afd_projection_contract.py` reports **3 passed**. The latter two need no
native runtime.

## Known uncovered scope

Recorded deliberately rather than papered over. None of these are gated today.

### Assertion depth

- Public latencies are compared **after** three-decimal rounding
  (`AFDInferenceSession._build_summary` rounds `ttft`/`tpot`/`request_latency`;
  `_combine_afd_static_estimate_results` rounds the same three in the AFD+PD path).
  `afd_layer_measurements` is the full-precision ingredient set, and this suite
  compares only its **key set**, never its values. A compensating change — attention
  faster, FFN slower, rounded TPOT unchanged — therefore passes.
- The latency ingredients in `.raw` are unpinned: `t_a_layer`, `t_f_layer`,
  `t_a2f_layer`, `t_f2a_layer`, `t_c_layer`, `t_step`, `balance_ratio`. A decode
  case's `.raw` carries 67 keys (`common.ColumnsAFD`, the DataFrame schema, lists 78
  — the two sets are not the same); the `path` assertions cover seven of them, ten
  for the two PD cases.
- In `both` mode the top-level `comm_hidden` is `None` **by design** (per-phase
  scalars are emitted instead), so the `both` case asserts nothing about
  communication hiding: `prefill_comm_hidden` / `decode_comm_hidden` are never
  checked. Communication hiding is only pinned for single-phase cases.
- Memory and OOM fields (`(a)memory`, `(f)memory`, `(a)is_oom`, `(f)is_oom`) and the
  power coverage gate are never asserted.

### The common configuration is not the product default

Defaults live in the `cli_estimate` signature. The baseline pins a different centre
point, so every case inherits at least one non-default value:

| Parameter | Baseline | `cli_estimate` default |
|---|---|---|
| `comm_overhead_factor` | 2.0 | 1.0 (also covered by `communication-one`) |
| `pipeline_model` | `conservative` | `optimistic` (also covered by cases) |
| `afd_combined_with_pd` | `false` | `true` (also covered by cases) |
| `a_batch_size` | 6 | 128 — no larger batch is exercised anywhere |
| `prefix` | 0 | 0 — never varied, though it drives `effective_prefill_len` |
| `nextn` | 0 | 0 — varied only by the strict-xfail defect test, see below |

Fixed at `None` and never varied: `attention_backend`, `gemm_quant_mode`,
`fmha_quant_mode`, `moe_quant_mode`, `comm_quant_mode`, `transfer_policy`. Of these,
`comm_quant_mode` changes bytes-per-element for all five AFD communication ops, so it
is materially load-bearing.

`batch_size` appears in the common query but is **dead for the 32 AFD-only cases**:
the AFD path reads `a_batch_size`, and only the static complement of the two
`*-with-pd` cases consumes `batch_size`.

### AFD + MTP (`nextn > 0`) is broken and pinned only as xfail

`cli_estimate` always builds a `SpeculativeDecodingProfile`. With `nextn = 0` the
projection early-returns, so the frozen matrix is safe. With `nextn > 0` and a
positive accepted-token count, the decode/static projection rebuilds the result dict
from a DataFrame constructed with `columns=common.ColumnsAFD`. That schema does
**not** contain `afd_layer_measurements`, so the key is dropped from
`EstimateResult.raw`.

Measured on this checkout with `nextn=1, nextn_accepted=1.0` over the baseline query:
`afd_phase="decode"` returns normally (`tpot = 24.502`) with the key **absent**, while
`afd_phase="prefill"` retains it because the projection early-returns for the prefill
role. The failure is therefore **silent at the API boundary** — no exception, just
missing data. Downstream, `aisimulate/sweeper/afd_perfmodel.py` raises
"AIC estimate did not return afd_layer_measurements".

This is a pre-existing product defect, not a defect in this suite, and it is
deliberately **not** fixed here. Two tests pin it: `test_afd_mtp_preserves_layer_measurements`
is a `strict=True` xfail asserting the key *should* survive, so an XPASS announces the
fix; `tests/test_afd_projection_contract.py` pins the schema coupling and needs no
native runtime. Retire both together, then add real AFD+MTP numerical cases.

### Validation coverage

These branches previously had no test anywhere in the repository. Each is now a named
rejection case pinning exception type, message pattern and the raising frame:

| Case | Branch |
|---|---|
| `invalid-ffn-ep-not-divisor` | `f_moe_ep_size` not a divisor of `f_tp_size` (`_run_afd_estimate`) |
| `invalid-zero-attention-nodes` | `n_a_nodes < 1`; `0` slips past the `is None` check in `cli_estimate` (`AFDConfig.__post_init__`) |
| `invalid-zero-a-batch-size` | `a_batch_size < 1` (`__post_init__`) |
| `invalid-zero-comm-overhead` | `comm_overhead_factor <= 0` (`__post_init__`) |
| `invalid-attention-tp-not-divisor` | `tp_a` not a divisor of `gpus_per_node`; the `< 1` half was already covered by `invalid-zero-attention-tp` (`__post_init__`) |
| `invalid-oom` | the AFD-side OOM `RuntimeError` (`_run_afd_estimate`) |

`f_moe_ep_size <= 0` needs no separate case: it short-circuits the same `or` condition
and raises the identical message from the identical frame, verified for `0` and `-1`.

Still uncovered: the **static-complement** OOM path in the combined AFD+PD branch, and
the `_validate_fp8_block_quantized_moe_config` rejection observed while probing for an
OOM configuration.

For contrast, these AFD rejections are covered elsewhere and are not gaps:
image/video `NotImplementedError` (`tests/unit/cli/test_afd_phase_completion.py`,
`tests/unit/sdk/task_v2/test_task_config.py`) and `forward_model="fpm"`
(`tests/test_afd_cli.py`).

Every expectation above was captured by executing the real entry point after
rebuilding the native runtime offline. No case in this file was written from source
reading alone.

## Runtime and CI notes

- **Collection path.** The Full CI `contracts` shard runs
  `pytest -q -c pytest.ini tests -n 4` from the repository root with no `-m`
  filter, so all three files here are collected. `scripts/select_full_ci.py` selects
  `application_tests` for any `tests/` or `scripts/` change.
- **Do not add `-c python/aisimulate/pytest.ini`.** The `gpu_0` and `pre_merge`
  markers are registered only in the root `pytest.ini`, and the project config also
  sets `--strict-markers`; running these files against it fails during collection.
- **`filterwarnings = error`** in the root `pytest.ini` is stricter than the project
  config's `ignore::DeprecationWarning`. Since this suite drives the *legacy* CLI
  entry point, any deprecation warning escaping that path becomes a test failure.
- **The three markers select nothing.** No CI job filters on `integration`, `gpu_0`
  or `pre_merge`; they are documentation. The tests run unconditionally in the
  contracts shard.
- **`scripts/` has no static gate.** The lint/format/compileall job targets
  `python/aisimulate/src/aisimulate`, two specific CLI test files, and `tests` —
  not `scripts`. `compare_afd_commits.py` is still imported and exercised by its
  guard tests, so logic regressions surface; style violations do not.
- **Building the native runtime.** The extension is a maturin/PyO3 module
  (`aisimulate._runtime`). A stale `_runtime.abi3.so` shows up as
  `AttributeError: module 'aisimulate_core._native' has no attribute ...` during
  collection, and it takes dozens of pre-existing root tests down with it. Rebuild
  with `uv sync --project python/aisimulate --extra dev` when PyPI is reachable; that
  command hangs rather than failing fast when it is not, so go straight to the offline
  path on a blocked network. Verified offline in this environment: `cargo metadata
  --offline` first confirms the dependency cache is complete, then `maturin build
  --offline --interpreter <venv python>` produces a wheel whose `_runtime.abi3.so` can
  be copied into `python/aisimulate/src/aisimulate/`. That file is gitignored, so
  replacing it is a local-only change; back it up first.
