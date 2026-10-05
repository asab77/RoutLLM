# FINAL held-out evaluation harness

FINAL is a one-time measurement. TRAIN supplied the predictor fit and grouped
model analysis; DEV supplied independent policy validation and exposed material
generalization limitations; FINAL is reserved only for measuring the already
frozen router. The router, threshold, candidates, evaluator, and headline
baseline must not change after FINAL is observed.

This experiment is an **offline held-out routing evaluation**, not production
traffic. It collects a complete candidate matrix, evaluates it, and then replays
the frozen router over persisted rows. There are 42 independent requests (six in
each of seven categories). The 168 request/model rows are repeated counterfactual
observations, so the statistical N is 42, not 168.

The predefined headline baseline is fixed Claude Sonnet 5. It was designated by
the TRAIN methodology as `ALWAYS_STRONGEST` before FINAL; it must not be replaced
with whichever FINAL candidate gives the largest apparent saving. All four fixed
models and the analysis-only retrospective oracle are reported alongside it.

## Frozen policy and operational authorization

`final-policy-1.0.json` freezes threshold 0.80, the exact portfolio, predictor
checksum, cheapest-qualifying selection, deterministic ties, fallback behavior,
serving-cost semantics, and the primary baseline. Its SHA-256 is:

`79da97dca54ffdff254122c7c0345a2136ebd7178cd18f8504757f4d0e31a8ba`

Protocol 1.7 remains byte-for-byte historical scientific evidence. Its stale
pre-confirmation wording is not rewritten. Operational readiness and human
authorization are separated into `final-execution-authorization.json`, which
references Protocol 1.7, current pricing evidence, and the successful Gemini
confirmation. The checked-in record intentionally remains
`AWAITING_INDEPENDENT_REVIEW`; paid execution fails closed until an independent
reviewer explicitly authorizes it. Historical `protocol.json` and
`identities.json` remain historical Protocol 1.4 records, not FINAL aliases.
Authorization requires named same-day human approval plus named same-day pricing
reverification; an older authorization fails closed. Paid execution additionally
requires a clean Git worktree and an authorization record bound to the exact
reviewed implementation commit and the complete outcome-critical source hash
set. That reviewed commit must be an ancestor of the clean execution-time HEAD;
every critical file must still be byte-identical. The run records the separate
execution-time commit. This permits a later authorization-only commit without a
self-referential commit hash. The checked-in authorization intentionally has no
implementation binding yet.

## Metrics

Primary task acceptability is `100 × acceptable selected responses / 42`.
Provider failures and unresolved/partial selected evaluations are non-acceptable
in this primary denominator. Reports additionally retain fully-evaluated coverage,
conditional acceptability among fully evaluated requests, provider failures,
judge failures, and unresolved counts. The metric is called task acceptability,
not general accuracy.

Serving cost includes only the realized candidate inference cost of the selected
row. It excludes semantic-judge cost, functional evaluation, other research
overhead, and the three counterfactual responses not selected by replay. Unknown
failed-attempt costs are never fabricated; incomplete cost coverage suppresses X
and the resume statement.

Summarization uses the frozen proposition Evaluator 1.3 with
`judge-gpt-6-astra`, one call per successful summary response and a 768-token
judge limit. Coding uses the pinned, network-disabled Docker functional sandbox.

DEV previously produced 33 acceptable selected responses among 41 valid selected
responses (80.49% conditional acceptability), including 1/6 reasoning responses.
The router was subsequently frozen without retraining. This context is retained;
FINAL remains measurement-only.

## Commands and mandatory checkpoints

All commands run from the repository root. Do not load credentials for dry runs.

### Stage 0 — non-paid preflight

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m adaptive_llm_gateway.evaluation.final_harness preflight
```

This hashes frozen inputs without parsing FINAL task content, checks evaluator
implementation hashes and pricing evidence, detects an existing completed run,
and makes zero provider calls. It does not authorize execution.

After independent review has updated the separate authorization record, the paid
gate must also pass:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m adaptive_llm_gateway.evaluation.final_harness preflight \
  --require-authorization --authorize-final
```

**STOP** on any hash, policy, portfolio, threshold, implementation, readiness,
authorization, or duplicate-run failure.

### Stage 1 — one-time candidate matrix

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m adaptive_llm_gateway.benchmarks \
  --dataset benchmarks/datasets/routing-benchmark-v1.2.json \
  --protocol benchmarks/protocols/routing-benchmark-v1.2/protocol-1.7.json \
  --pricing-readiness benchmarks/protocols/routing-benchmark-v1.2/execution-readiness-1.7.json \
  --split final \
  --split-manifest benchmarks/protocols/routing-benchmark-v1/split-manifest.json \
  --allow-final-evaluation --authorize-final \
  --final-policy benchmarks/protocols/routing-benchmark-v1.2/final-policy-1.0.json \
  --final-authorization benchmarks/protocols/routing-benchmark-v1.2/final-execution-authorization.json \
  --final-results-root artifacts/routing-benchmark-v1/final-runs \
  --models candidate-nemotron-3.5-lightning candidate-gpt-6-luna candidate-gemini-3-flash candidate-claude-sonnet-5 \
  --limit 42 \
  --output artifacts/routing-benchmark-v1/final-runs \
  --allow-paid
```

Expected and hard workflow limits are 168 candidate attempts and zero retries.
Each task/model attempt is claimed in an experiment-wide durable ledger before
the provider boundary and its result or failure is persisted immediately. A new
run ID reuses terminal entries and cannot reset the budget. A `started` entry
without a terminal result is ambiguous and fails closed; it is never retried
automatically.
**STOP** if the run is not exactly 42 requests × four candidates, if storage
fails, or if any frozen configuration snapshot differs.

### Stage 2 — complete Evaluator 1.3 plan

First confirm the zero-call plan:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m adaptive_llm_gateway.evaluation.final_harness semantic-judge \
  --root artifacts/routing-benchmark-v1/final-runs \
  --run-id <RUN_ID> --dry-run
```

### Stage 3 — one-time complete evaluation

Only after the dry run and authorization gate pass:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m adaptive_llm_gateway.evaluation.final_harness semantic-judge \
  --root artifacts/routing-benchmark-v1/final-runs \
  --run-id <RUN_ID> \
  --execute --allow-paid-judge --authorize-final
```

This binds all frozen deterministic evaluators, Evaluator 1.3, the proposition
specification, Astra, the 768-token judge limit, and the functional sandbox in
one complete evaluation pass. It performs at most 24 coding sandbox executions,
makes at most 24 judge calls, and has no retries. The command writes a durable
per-task/model semantic ledger entry immediately before each judge call and
persists the sanitized evaluation immediately afterward. Completed calls are
reused on restart. Ambiguous `started` entries fail closed and require independent
reconciliation; never delete or edit the ledger to force a retry. The exact
digest-pinned sandbox image, network isolation, read-only/non-root execution,
capability restrictions, security options, resource limits, and timeout are all
validated before the semantic ledger is initialized.

### Stage 4–7 — export, replay, aggregation, canonical results

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m adaptive_llm_gateway.evaluation.final_harness replay \
  --root artifacts/routing-benchmark-v1/final-runs \
  --run-id <RUN_ID> \
  --output artifacts/routing-benchmark-v1/final-results/<RUN_ID> \
  --authorize-final
```

Replay makes zero provider calls. It verifies the predictor and policy, emits one
selection per request, aggregates RoutLLM and all fixed models, labels the oracle
analysis-only, and writes `final-results.json` plus `final-report.md`. The resume
statement is suppressed when cost is incomplete, N is not 42, category N is not
six, or identity verification is not ready. Canonical JSON is the publication
source of truth. If a process stops after JSON is atomically published but before
Markdown is published, rerunning replay reconciles the missing Markdown from the
validated JSON without rerunning inference, judging, functional evaluation, or
routing.

## Protected-data test classification

Do not run the nine legacy modules below wholesale. The safe nodes use only
protocol metadata, hashes, mocks, or synthetic objects; the unsafe nodes parse a
protected routing dataset or invoke a dry-run path that loads it.

- `test_corrected_pilot_preflight.py`: all nodes are safe except
  `test_corrected_pilot_dry_run_reaches_paid_boundary_without_crossing_it`.
- `test_gemini_budget_correction.py`: all nodes are safe except
  `test_protocol_15_pricing_and_confirmation_dry_run_are_strict`.
- `test_gemini_minimal_correction.py`: protocol semantics, fallback, mocked
  transport, and artifact-hash nodes are safe; allowance/protected-hash and
  readiness/cost-plan nodes are unsafe.
- `test_gemini_native_minimal_protocol.py`: protocol semantics and artifact-hash
  nodes are safe; readiness/allowance/cost nodes are unsafe.
- `test_protocol_split_execution.py`: the first split-selection parametrization
  parses the protected dataset. The remaining model-contract and unsupported
  protocol nodes are safe.
- `test_routing_benchmark_v1.py`: synthetic construction and validation nodes are
  safe; `test_canonical_hashes_match_written_artifacts` and any node changed to
  load a checked-in routing dataset are unsafe.
- `test_routing_benchmark_v12_selector.py`: nodes using the module-level
  `dataset_v12` fixture are unsafe. The Protocol 1.7 identity/contract node is
  safe.
- `test_summarization_proposition_spec.py`: schema-only material-error tests are
  safe. Specification, validation-set, dataset, and historical-artifact nodes
  load protected benchmark material and are unsafe.
- `test_phase9_protocol_correction.py`: typed allowance, bounded headroom,
  deterministic policy, and semantic-boundary fixture nodes are safe. Dataset,
  stored evidence, and corrected rerun-identity nodes are unsafe.

`tests/test_final_harness.py` is the protected-data-free regression entry point
for the FINAL harness. It constructs all tasks, results, ledgers, repositories,
and Git histories synthetically.

## Call and cost budget

- Candidate calls: exactly 168; expected $0.08225315, estimated maximum $0.19935755.
- Astra calls: at most 24; expected $0.71932000, estimated maximum $1.21212000.
- Combined expected: $0.80157315.
- Repository-estimated maximum: $1.41147755.
- Conservative budget with 10% contingency: $1.55262531.

Pricing must be independently rechecked immediately before authorization. No
FINAL result is claimed by this document.
