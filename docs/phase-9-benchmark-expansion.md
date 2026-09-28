# Phase 9 Part A: Benchmark expansion and router evidence scale-up

Status: **design proposal; human approval required before dataset generation or paid execution**

## 1. Motivation

RouteLLM now has a working production routing path and useful grouped out-of-fold
ranking evidence. The remaining limitation is experimental scale. Foundation V3
contains 56 independent requests and 224 request/candidate attempts. Phase 8G
found that this supports descriptive routing analysis, but does not support a
stable global threshold, probability guarantee, or frozen quality-mode policy.

Phase 9 should increase the number and structural diversity of independent
requests, preserve a genuinely untouched final test, and measure routing quality,
cost, and calibration without changing production behavior.

## 2. Existing architecture audit

The following components can be reused unchanged:

- `BenchmarkTask` and `BenchmarkDataset` provide typed task content, stable task
  IDs, source-byte SHA-256 identity, output limits, category, difficulty metadata,
  and evaluator metadata.
- `BenchmarkRunner` validates models before execution, snapshots request features,
  effective output limits, prices, capabilities, provider pins, and protocol data,
  then executes each task/model pair without a production telemetry sink.
- `FileBenchmarkRepository` writes the run manifest before provider work, records
  each result atomically, and preserves completed/aborted status.
- The evaluation repository keeps candidate generation and evaluation artifacts
  separate from production telemetry.
- Accepted-answer, classification, extraction, structured-JSON, reasoning,
  functional-code, and semantic-summary evaluators already preserve explicit
  evaluation statuses and missing labels.
- Coding uses restricted functional execution with declared fixtures; summary
  evaluation combines deterministic checks with a blind, structured semantic
  rubric and factual-consistency veto.
- Routing export preserves frozen candidate configuration, authoritative usage,
  costs, evaluator provenance, valid/missing label status, and pre-generation
  request features.
- Phase 7 grouped folds, Phase 8C-0 no-provider-pin formulation, and Phase 8G
  calibration analysis provide reusable metric and policy-analysis code.
- Existing dataset, protocol, run-manifest, evaluation, and routing-export hashes
  provide a good provenance base.

The expansion needs four additions in a later implementation phase:

1. task-family and source-provenance metadata;
2. family-aware dataset validation and deterministic split manifests;
3. a preflight duplicate/diversity/cost audit that runs before paid inference;
4. an operationally sealed final-test artifact and explicit final-evaluation
   command.

These additions belong to benchmark tooling. They do not require changes to the
production router, predictor artifact, HTTP APIs, provider execution, or telemetry.

## 3. Candidate dataset sizes

All options retain seven balanced categories and four frozen candidates. The
objective-evaluator count covers six categories. Semantic-judge count assumes all
four responses for every summarization request are successfully generated and
judged exactly once.

| Option | Requests/category | Independent requests | Candidate attempts | Objective evaluations | Semantic-judge calls |
|---|---:|---:|---:|---:|---:|
| Small | 25 | 175 | 700 | 600 | 100 |
| Medium | 32 | 224 | 896 | 768 | 128 |
| Large | 40 | 280 | 1,120 | 960 | 160 |

### Cost basis

The local configuration freezes candidate prices verified on 2026-09-24:

| Candidate | Input USD/1M | Output USD/1M |
|---|---:|---:|
| Nemotron 3.5 Lightning | 0.05 | 0.15 |
| GPT-6 Luna | 0.10 | 0.50 |
| Gemini 3 Flash | 0.50 | 3.00 |
| Claude Sonnet 5 | 2.00 | 10.00 |

Astra is configured at $10/1M input tokens and $50/1M output tokens. The completed
Foundation V3 run cost $0.07364335 for 56 requests across all candidates and
$0.29840 for 31 completed Astra judgments. These observations imply:

- candidate baseline: $0.00131506 per independent request across four candidates;
- judge baseline: $0.00962581 per semantic judgment.

Planning estimates multiply both baselines by 1.25 for broader/longer tasks, then
add a separate 10% execution contingency. This is a budget model, not a promise.
Prices and catalog availability must be reverified immediately before a paid
pilot.

| Option | Candidate estimate | Judge estimate | Subtotal | 10% contingency | Planning budget |
|---|---:|---:|---:|---:|---:|
| Small | $0.2877 | $1.2032 | $1.4909 | $0.1491 | **$1.64** |
| Medium | $0.3682 | $1.5401 | $1.9083 | $0.1908 | **$2.10** |
| Large | $0.4603 | $1.9252 | $2.3854 | $0.2385 | **$2.63** |

Foundation V3 observed sequential provider plus judge latency was about 9.9
minutes. Linear scaling with the same 25% allowance gives approximately 39, 50,
and 62 minutes of provider wait time for small, medium, and large. Local generation,
human review, validation, and failure diagnosis will dominate elapsed project time.

### Recommendation

Use the **medium expansion: 32 new requests per category, 224 total, 896 candidate
attempts, and a $2.10 planning budget**. It provides 4x as many fresh requests as
Foundation V3, supports balanced 140/42/42 train/development/final partitions, and
avoids the additional review burden of 280 tasks until the pilot demonstrates that
the new design is discriminative and reliable. The large option remains a later
extension if uncertainty is still too wide.

## 4. Dataset composition

### Category distribution

The medium design contains exactly 32 new requests in each existing category:

| Category | Requests | Candidate attempts | Primary evaluator |
|---|---:|---:|---|
| Classification | 32 | 128 | Exact normalized label |
| Coding | 32 | 128 | Restricted functional execution |
| Extraction | 32 | 128 | Structured JSON field F1 |
| Structured JSON (`json` internally) | 32 | 128 | Syntax, schema, and value score |
| QA | 32 | 128 | Accepted-answer exact match |
| Reasoning | 32 | 128 | Normalized final-answer match |
| Summarization | 32 | 128 | Deterministic checks plus semantic judge |

No new production category is introduced. The persisted internal category remains
`json`; reports may display it as `structured_json`.

### Difficulty distribution

Per category, author **8 easy, 13 medium, and 11 hard** tasks. Across 224 requests
this yields 56 easy (25.0%), 91 medium (40.6%), and 77 hard (34.4%). The medium and
hard share produces useful disagreement while preserving enough easy tasks to
detect regressions and floor effects.

Difficulty is evaluation-only metadata and must never enter production features.
Assign it from a category-specific rubric before candidate execution:

- easy: one primary operation, explicit output contract, few interacting facts or
  constraints, and no plausible competing interpretation;
- medium: two to four dependent operations or discriminations, meaningful
  distractors, and at least one interaction between facts, rules, or constraints;
- hard: four or more material operations where appropriate, several interacting
  constraints, plausible distractors/failure modes, and one machine-verifiable or
  explicitly rubric-governed answer.

Task length, large numbers, and cosmetic wording do not increase difficulty by
themselves. Two reviewers should be able to reproduce the assigned level from the
rubric; disagreements block execution until resolved.

## 5. Task-family design and independence

Add evaluation-only metadata with these semantics:

| Field | Meaning |
|---|---|
| `task_family_id` | Stable identifier for a shared reasoning/structural template |
| `source_type` | `human_authored`, `deterministic_generator`, or later approved source |
| `source_id` | Versioned authoring recipe, generator, or review record |
| `generation_seed` | Seed for deterministic generation, otherwise absent |
| `evaluation_type` | Explicit evaluator contract selected before execution |
| `family_variant` | Human-readable structural variation within a family |

The family ID is benchmark metadata only. It must not appear in production request
features or the router artifact.

Independence rules:

- Changing names, numbers, entities, variable names, ordering, or prose while
  retaining the same solution structure does not create a new family.
- No family contributes more than two tasks to the medium expansion without a
  documented exception approved during dataset audit.
- Aim for at least 16 families per category and at least five families per category
  in the final test.
- Multi-task families are assigned wholly to one split.
- A family definition records its invariant reasoning structure and the dimensions
  that are allowed to vary.
- Foundation V2/V3 tasks receive retrospective family annotations before overlap
  checks. A new task structurally matching a Foundation family is not eligible for
  development or final test.

## 6. Anti-duplication and leakage control

Run these transparent checks before the dataset can be frozen:

1. Unicode-normalize, case-fold, collapse whitespace, and hash exact task text plus
   output contract; reject exact duplicates.
2. Canonicalize task content by replacing quoted identifiers, numeric literals,
   and declared entity slots; use this as a diagnostic for cosmetic variants.
3. Compute token 3-gram and 5-gram Jaccard similarity and normalized edit similarity.
   Flag, rather than automatically reject, near matches for family review.
4. Require every task to have a reviewed family ID and family invariant.
5. Compare new content and family IDs against Foundation V1/V2/V3 and every new
   split, including held-out content.
6. Record all flags and reviewer dispositions in the provenance manifest.

No embedding or provider API is needed for initial duplicate detection. If later
evidence shows deterministic diagnostics miss structural duplicates, an offline
local method may be proposed separately.

Split assignment operates on `task_family_id`, not candidate rows or request IDs.
The same request and all four candidate results always remain together. Preprocessing,
feature selection, calibration, threshold selection, and policy design must fit
without final-test families.

## 7. Train, development, and final-test strategy

For the recommended 224 new requests, use this category-balanced target:

| Split | Per category | New requests | Purpose |
|---|---:|---:|---|
| Train | 20 | 140 | Fit predictors; grouped CV for formulation/model development |
| Development | 6 | 42 | Threshold, calibration, and policy analysis |
| Final test | 6 | 42 | One frozen generalization evaluation after policy freeze |

Assignments are deterministic from a versioned split seed and family ID, then
audited for category and difficulty balance. Exact 20/6/6 counts are targets;
family integrity takes precedence, with any deviation recorded in the manifest.
No outcome label may be used to choose the split.

Continue grouped cross-validation inside the train split, grouping by family and
request. Use it for predictor comparison, feature decisions, and any later
hyperparameter choice. Development data may compare policy thresholds and
calibration approaches after the predictor formulation is chosen. Once those
choices are frozen, a final training recipe may refit on train plus development
without inspecting final labels, followed by one final-test evaluation.

### Foundation V3 role

Foundation V3 is repeatedly inspected historical evidence and cannot be a pristine
final test. Annotate its families and use non-overlapping V3 requests as
**historical training-only data**. Keep separate provenance so results can also be
reported with and without V3. Any V3 family overlapping new development or final
families remains historical-reference-only and is excluded from fitting. V3 must
not select thresholds, calibration, or final policy.

This gives up to 196 training requests (140 fresh plus 56 historical) while keeping
42 fresh development and 42 fresh final requests.

## 8. Category diversity requirements

### Classification

Vary single-label taxonomies, hierarchical precedence, mutually plausible labels,
exception rules, abstention/insufficient-information cases, multi-record decisions,
and policies where one subtle condition changes the label. Avoid label inference
from keyword matching. Historical Gemini classification failures caused by
configuration are not capability evidence.

### Coding

Cover sequence/string transformation, interval and scheduling logic, aggregation,
state machines, parsing, validation, graph/tree traversal, dynamic programming,
and data-structure behavior. Vary signatures, edge cases, mutation requirements,
ordering guarantees, and error contracts. Every hidden fixture must follow directly
from the prompt, and oracle implementations must pass independently authored tests.

### Extraction

Vary prose, tables, semi-structured logs, nested records, repeated fields,
conflicting mentions resolved by explicit rules, optional/null fields, units,
ordering, and multi-entity linkage. Expected JSON paths and normalization rules
must be explicit.

### Structured JSON

Vary nested schemas, filtering, aggregation, sorting, conditional inclusion,
cross-field invariants, nullability, type distinctions, and exact-key constraints.
Separate valid-JSON behavior from correct schema and values.

### QA

Use context-grounded synthesis across multiple facts, temporal updates, entity
resolution, controlled comparisons, set intersections, and rule-conditioned
answers. The answer must be uniquely determined by supplied context; external
knowledge and ambiguous directionality are forbidden.

### Reasoning

Use original ordering, scheduling, assignment, constraint-satisfaction, finite
state, logic-grid, resource-allocation, and counterfactual problems. Programmatically
enumerate solutions where practical and require exactly one accepted result.

### Summarization

Vary narrative, status report, incident report, policy memo, multi-speaker notes,
technical explanation, and mixed relevant/distractor text. Vary compression ratio,
required facts, chronology, attribution, uncertainty, forbidden claims, word/sentence
limits, and required framing. Semantic requirements must identify what is necessary
without encoding preferred prose.

Foundation V3 disagreement was strongest in coding, extraction, structured JSON,
reasoning, and summarization. Classification contained configuration-induced
missing labels and several all-pass tasks; QA often produced broad failure rather
than useful separation. The expansion should improve classification policy nuance
and QA answerability, while preserving discriminative structures in the other five
categories. It must not merely oversample old failures.

## 9. Discriminative-task design

Within each category, target a planned mix of task structures rather than known
model outcomes:

- about 25% straightforward competence/regression tasks likely solvable by all;
- about 50% tasks with meaningful structure where cheap and medium models may
  differ;
- about 20% capability-intensive tasks where stronger modeling may matter;
- at most about 5% boundary cases where all models may fail.

These are authoring targets, not label quotas. Candidate outcomes must never be
edited to force the desired distribution. If the pilot shows saturation or universal
failure, revise task families and rerun local validation before any full execution.

## 10. Evaluator and ground-truth plan

| Category | Reuse | Required extension/audit | Ground truth |
|---|---|---|---|
| Classification | Normalized exact label | Validate declared label vocabulary and output-only contract | One versioned label plus policy derivation |
| Coding | Restricted functional sandbox | Add fixture/oracle self-tests for every new family; confirm resource bounds | Oracle function plus public/hidden cases |
| Extraction | JSON field F1 | Declare list-order and optional-field semantics per task | Canonical JSON and normalization contract |
| Structured JSON | Syntax/schema/value evaluator | Add explicit allowed/required/extra-key and ordering rules where needed | Canonical typed JSON and schema assertions |
| QA | Accepted-answer match | Ensure all semantically equivalent concise forms are enumerated or deterministically normalized | Unique answer and derivation |
| Reasoning | Final-answer match | Programmatic uniqueness verification where possible | Unique result, accepted forms, solver audit |
| Summarization | Deterministic checks plus blind Astra rubric | Revalidate judge on representative new structures before pilot | Required facts, forbidden/unsupported claims, constraints, rubric |

Evaluator extensions must precede dataset freeze. They cannot weaken correctness to
admit ambiguous tasks. Every objective task must pass a positive oracle self-test
and at least one mutation/negative test. Every summary must include explicit source
facts, semantic requirements, unsupported-claim guidance, and deterministic output
constraints.

Candidate/provider failures and evaluator infrastructure failures remain missing
labels. They are never imputed as negative. Configuration failures are separately
classified, diagnosed, and excluded from capability claims.

## 11. Semantic-judge cost control

- Use Astra only for summarization tasks that genuinely require semantic judgment.
- Run deterministic format and content-safety prechecks first. A deterministic
  veto may avoid a judge call only when the persisted evaluation contract declares
  the result conclusively unacceptable; this rule must be frozen before execution.
- Cache only by the complete immutable tuple of dataset hash, task ID, candidate
  response hash, judge model/configuration, prompt version, rubric version, and
  evaluator version. Cache reuse must be recorded; no response may be judged twice.
- Keep the judge blind to candidate identity, price, latency, token usage,
  difficulty, and prior outcomes.
- Revalidate structured parsing, factual-consistency veto, usage, and cost
  accounting on the pilot before approving the full run.
- Preserve real semantic evaluation; do not substitute keyword coverage for cost.

The medium design expects at most 128 judge calls before any valid deterministic
short-circuit. The budget assumes all 128 calls, so savings do not need to be relied
upon.

## 12. Token-budget strategy

Start from Foundation V3 task-level requirements, but assign each new task an output
allowance from its expected response contract:

- classification: 32 tokens for label-only output; larger only when the declared
  contract requires structured rationale;
- QA: 64 for concise answers, up to 160 for explicitly required synthesis;
- extraction: 64–192 based on maximum canonical JSON size;
- structured JSON: 64–192 based on maximum canonical schema instance;
- reasoning: 160 by default, with capability-policy overrides such as Gemini's
  typed category allowance resolved through `ModelConfig`, never model-slug cases;
- coding: 128–256 based on oracle implementation and required signature;
- summarization: 64–192 based on source length and explicit word/sentence limit.

For each task, serialize the canonical expected output or oracle and verify that its
token estimate fits comfortably below the limit. The pilot must measure truncation
and visible/reasoning token use. Any candidate-specific allowance must be justified
by typed capability policy and frozen in the inference protocol; no task runner
model-slug hacks are allowed.

## 13. Dataset-generation strategy

Use three controlled sources:

1. human-designed family specifications describing invariants, allowed variation,
   ground-truth construction, and failure modes;
2. deterministic generators for finite structured problems whose answers can be
   solved and uniqueness-checked programmatically;
3. carefully authored individual tasks for semantic/contextual cases that do not
   fit safe generators.

Generation must not use candidate outputs. If an LLM-assisted authoring process is
proposed later, record its model and prompt provenance, treat its output as an
untrusted draft, independently construct/verify ground truth, and charge generation
cost outside the execution budget. No LLM generation is approved by this document.

Generated candidates enter a local staging dataset, pass deterministic validation,
duplicate/family review, evaluator self-tests, and human review, then receive a
frozen identity. Dataset generation and paid execution remain separate commands
and approval checkpoints.

## 14. Data-quality gates

A task is ineligible for paid execution unless all gates pass:

- schema, dataset identity, category, difficulty, output type, and task ID valid;
- prompt and system contract non-empty and unambiguous;
- unique normalized content and reviewed family metadata;
- no unreviewed near-duplicate flags or family overlap across splits;
- evaluator type/version and acceptance threshold explicit;
- complete, versioned ground truth and derivation;
- objective evaluator positive and negative self-tests pass where applicable;
- reasoning/coding generator confirms uniqueness and fixtures where applicable;
- semantic requirements and factual-consistency contract complete;
- canonical expected output fits the declared token budget;
- candidate configuration, prices, provider pins, and typed output policy frozen;
- deterministic split manifest has no request/family leakage;
- prompt/content security scan passes and contains no credentials or private data;
- dataset, evaluator, split, and protocol hashes agree with the preflight manifest;
- pre-run cost remains within the explicitly approved budget.

Validation fails closed before repository initialization or provider work.

## 15. Dataset identity and manifests

Name the expansion **RouteLLM Routing Benchmark v1**, with machine identity
`routellm-routing-benchmark-v1` and semantic version `1.0.0`. Do not call it
Foundation V4: this is a family-aware experimental suite with train/development/
final partitions, rather than a direct immutable revision of Foundation V3.

Freeze these version-controlled artifacts before the pilot/full run:

- canonical dataset or split-specific task files;
- dataset content SHA-256;
- generation/provenance manifest;
- family manifest;
- deterministic split manifest and seed;
- evaluator-version and ground-truth manifest;
- candidate and semantic-judge configuration manifest;
- inference-protocol hash;
- duplicate/audit report;
- pre-run cost report.

Any content, split, evaluator, candidate, or protocol change increments the relevant
version and invalidates downstream run compatibility.

## 16. Final-test freeze safeguards

- Materialize final tasks and labels in a separate version-controlled artifact
  whose normal training/development loaders reject the `final` role.
- Training and policy commands accept only train/development manifests and verify
  their hashes.
- Require an explicit `final-evaluate` command, frozen predictor/policy hash, and a
  human approval record before loading final labels.
- Reports before final evaluation may show final task counts/families but not labels,
  candidate outcomes, or metric previews.
- Run final evaluation once for the frozen decision. Any later rerun is a new
  reported evaluation event, never silently replacing the first.
- After final labels are inspected, that release remains historical and cannot be
  reused as a pristine final set for another policy iteration.

These controls reduce accidental leakage without claiming cryptographic secrecy.

## 17. Pilot design

Run a **21-request paid pilot: one easy, one medium, and one hard request per
category**, drawn only from families assigned to the future training split. This is
84 candidate calls, 12 maximum Astra calls, and about $0.20 under the same conservative
cost model. Pilot results may improve tooling, task wording, evaluators, and token
budgets, but the modified tasks must be versioned and revalidated. Pilot tasks never
move into development or final test.

The pilot must cover:

- all seven categories, all difficulty levels, and at least two source types;
- every evaluator path, including functional execution and semantic judging;
- representative small and large response contracts;
- candidate configuration, provider pins, reasoning settings, and typed output
  allowances;
- artifact persistence, usage, latency, cost, missing labels, and security behavior.

### Pilot stop rules

Stop before paid calls if any schema, family, split, ground-truth, evaluator,
configuration, hash, secret scan, or cost check fails. Stop during/after the pilot
before further calls when any of these occurs:

- a systematic candidate configuration or provider-pin error;
- ambiguous ground truth or a functional fixture not entailed by its prompt;
- semantic judge parsing, blindness, or factual-consistency-veto failure;
- truncation indicates an under-budgeted task family;
- duplicated/family leakage is discovered;
- actual spending exceeds the approved pilot cap or materially exceeds the model;
- infrastructure failures affect more than one task/candidate pair or show a common
  cause that must be diagnosed first;
- artifacts, usage, or costs cannot be reproduced from the run manifest.

No automatic retry is part of the pilot. A retry policy, if needed, requires a new
human-reviewed protocol.

## 18. Full-run design and stopping rules

After pilot review, regenerate the cost model from measured tokens and latency,
freeze the full manifests, and request separate human approval. Execute tasks and
models in deterministic order with atomic per-result persistence. Objective
evaluation and semantic judging remain separate, reproducible stages.

Do not start or continue the full paid benchmark if:

- unresolved exact, near-duplicate, or cross-split family overlap exists;
- evaluator ambiguity, oracle disagreement, or ground-truth validation remains;
- candidate/judge configuration, pricing, catalog availability, provider pins,
  reasoning settings, or token policy is not frozen;
- expected total cost exceeds the approved budget;
- any dataset, evaluator, split, family, or protocol hash differs from approval;
- semantic judge validation or artifact storage fails;
- final-test isolation cannot be demonstrated;
- pilot evidence shows widespread saturation, universal failure, configuration
  failure, or token truncation that undermines the design.

Execution checkpoints are mandatory:

`design approval → local generation → dataset audit → pilot estimate → human
approval → paid pilot → pilot review → full estimate → human approval → full run`.

No paid stage automatically triggers another.

## 19. Future router experiment

No training occurs in Part A. After collection, compare under identical family-aware
splits:

1. accepted `INTERACTION_NO_PROVIDER_PIN` logistic model;
2. additive logistic baseline;
3. Rule-Based V1;
4. always cheapest;
5. matched always strongest;
6. retrospective cheapest-acceptable oracle, clearly analysis-only.

Fit preprocessing only on training families. Use grouped CV inside training for
model decisions, development for threshold/calibration/policy selection, then freeze
the complete recipe before final evaluation. Report metrics with and without
historical Foundation V3 training rows.

Consider one small gradient-boosted tree model only if all conditions hold:

- at least roughly 175 fresh training requests remain after family grouping;
- logistic residual analysis shows stable nonlinear interactions across folds;
- the nonlinear model improves proper losses and within-request ranking in grouped
  CV and development, rather than one metric or category;
- calibration and feature-attribution behavior remain auditable;
- no final-test feedback informs the decision.

Neural networks remain unjustified at this scale.

## 20. Success metrics and decision rules

Define success as comparison-based evidence rather than an arbitrary pass score.

### Prediction

- grouped log loss and Brier score versus additive logistic and historical model;
- ROC-AUC and average precision with grouped uncertainty;
- within-request pairwise ranking and top-1 acceptable rate;
- category and candidate diagnostics with visible sample counts.

### Routing

- acceptable selection rate and mean quality on identical valid-label coverage;
- average/total cost and cost reduction against matched always strongest;
- fallback rate, threshold-met rate, and selected-model distribution;
- comparison against always cheapest, always strongest, and Rule-Based V1;
- grouped bootstrap intervals and sensitivity to missing labels.

The learned policy should improve quality over always cheapest and reduce cost
against always strongest without a practically important quality loss. It should
justify its complexity against Rule-Based V1 on untouched development and final
requests. Exact acceptable tradeoffs must be approved before final evaluation.

### Calibration and stability

- equal-width and equal-frequency reliability tables, ECE, MCE, slope/intercept,
  and proper losses;
- candidate/category diagnostics treated according to independent sample size;
- threshold stability, cost/quality frontier, and grouped bootstrap uncertainty;
- post-hoc calibration only when cross-fitted and demonstrably better on development.

### Generalization

The primary claim comes from the untouched family-isolated final test. Report the
frozen policy once with confidence intervals and all missing-label/configuration
caveats. Do not use final results to retroactively choose a threshold or model.

## 21. Expected limitations

Even 224 new requests remain modest for candidate-by-category calibration and rare
failure analysis. Four candidate rows from one request are correlated, so the
effective sample size is requests/families, not 896. Family definitions require
human judgment. Semantic labels retain judge variance. Provider catalogs and prices
can change between design and execution. A 42-request final test yields much better
evidence than Foundation V3 reuse, but still produces wide intervals for small
quality differences and category-specific claims.

These limitations favor transparent grouped uncertainty and comparison-based policy
decisions over precise-looking global guarantees.

## 22. Production boundary

Phase 9 Part A changes only this design document. It does not modify dataset files,
the benchmark runner, evaluators, production features, predictor formulation,
candidate configuration, APIs, services, thresholds, provider execution, response
validation, escalation, telemetry, Redis, dashboard, or deployment behavior. It
makes no provider or semantic-judge calls and performs no training.

## 23. Implemented benchmark construction

Phase 9 local construction implements the approved design as **RouteLLM Routing
Benchmark v1** (`routellm-routing-benchmark-v1`, version `1.0.0`). This section
records implementation facts separately from the approved design above. It does
not report candidate or routing outcomes.

- The canonical dataset contains 224 new tasks: 32 in each category.
- Every category contains 8 easy, 13 medium, and 11 hard tasks. Overall counts are
  56 easy, 91 medium, and 77 hard.
- There are 16 families per category and 112 families overall. Every family has
  exactly two tasks and remains wholly inside one split.
- The deterministic split contains 140 train, 42 development, and 42 protected
  final-test tasks, exactly 20/6/6 per category.
- Controlled programmatic construction is recorded for every task with stable
  family, source, seed, evaluator, and family-variant metadata. Candidate outputs
  and external LLM generation were not used.
- Objective evaluator self-tests passed for 160 non-coding tasks. All 32 QA
  normalization checks and 32 programmatic reasoning oracles passed.
- The pinned Docker sandbox accepted all 32 canonical coding solutions and rejected
  all 32 representative incorrect solutions.
- All 32 summarization contracts passed local source, requirement, constraint,
  token-budget, and frozen judge-contract validation without calling Astra.
- Duplicate auditing performed 57,904 within-dataset and historical comparisons.
  It recorded 46 deterministic dispositions and left zero unresolved exact,
  canonical, or suspicious near-duplicate findings.
- The pilot manifest selects 21 train-only tasks: one easy, medium, and hard task
  from distinct families in each category. It represents 84 candidate calls and at
  most 12 future judge calls. It has not been executed.
- Normal benchmark execution requires an explicit split. Final-test execution is
  rejected unless an explicit final-evaluation gate and 64-character predictor and
  policy identities are supplied. Inspecting later final outcomes consumes pristine
  status for this release.

Canonical identities:

| Artifact | SHA-256 |
|---|---|
| Dataset content | `1d79b3f28f15321730b42c6bb17a3e37b97eaaa7b65cd99d945e564cd33a5d4c` |
| Split manifest | `ada872f3653c33f1e6dab721756094bbf1ed09a17b518b6593ae0bc5a32d9711` |
| Evaluator manifest | `2631014ab4e2f0b2b621cd45b63aca20c7accbf64dcf6e56c2d6ff531228eefb` |
| Provenance manifest | `cf4c4d831112d84e40ffaf5aea680bd365db559da6f8d2c7c3b6c8a9809a91d4` |
| Benchmark protocol | `5b8be0635b2c10b87603930493bc8158d5b6f89be8530636119baa1b2801063b` |

The task-specific local estimate expects 36,156 candidate input tokens, 53,159
candidate output tokens, and a candidate cost of `$0.21072690` for the full run.
It expects 128 Astra calls costing `$1.00625000`; with 10% contingency, the full
planning total is `$1.33867459`. The 21-task pilot estimate is `$0.12455294`, also
including contingency. Both are below the earlier conservative `$2.10` and `$0.20`
design allowances because the constructed prompts and response contracts permit
more precise task-specific estimates.

Pricing remains `REQUIRES_REVERIFICATION`. A human must review task content,
duplicate dispositions, protocol configuration, and refreshed pricing before any
paid pilot. No paid execution, pilot, full benchmark, model training, threshold
selection, or final-test evaluation is authorized by this implementation.

## Content revision 1.1.0

Human review rejected the initial 1.0.0 content for systematic difficulty labeling,
cosmetic coding and summarization variants, synthetic QA/reasoning wording, and a
weak pilot sample. Version 1.1.0 preserves the benchmark infrastructure and replaces
the generation methodology:

- family index controls only the group-safe train/development/final split;
- category-specific capability scores determine difficulty independently of split;
- every coding family has two distinct behaviors with independently useful fixtures;
- every summarization family has independently authored sources and fact contracts;
- QA uses explicit answer-only contracts with deterministic normalization, and the
  malformed 48-hour policy task is corrected;
- reasoning scenarios use operational scheduling, dependency, allocation, and state
  contexts where possible;
- the content-quality gate requires zero cosmetic families, malformed prompts,
  pilot trivial-only tasks, coding review items, and summarization review items.

The redesign retains 224 tasks, 32 per category, 16 families per category, the
8/13/11 per-category difficulty distribution, and the 140/42/42 group-safe split.
New identities and cost estimates are generated by the canonical offline build;
pricing remains `REQUIRES_REVERIFICATION` until the approved pre-paid-call check.

## Protocol correction 1.2.0

The 1.1.0 paid pilot showed that a task's visible-answer budget was being sent
unchanged as the provider generation allowance. For reasoning-enabled paths, the
provider counted hidden reasoning and visible output inside that single allowance.
This produced seven empty length-terminated responses and additional incomplete
visible fragments.

Protocol 1.2.0 keeps every task and its visible-output requirement unchanged. A
benchmark-only typed model capability now declares whether the upstream allowance
counts visible output alone or reasoning plus visible output. Combined-accounting
models receive a deterministic additive reserve: 128 tokens for Luna, 256 for
Gemini, and 128 for Sonnet. Nemotron keeps the canonical visible requirement. The
reserve is bounded, independent of category, and resolved without model-slug
branches. The production candidate registry and Phase 7/8 feature contract remain
unchanged.

The semantic judge contract is independently versioned as 1.1.0. It now states
that source-entailed semantic compression is a supported paraphrase, while new or
strengthened causes, motives, diagnoses, quantities, events, actors, attributions,
certainty, and operational consequences remain unsupported. Twelve local boundary
fixtures cover causal wording, diagnosis, attribution, certainty, temporal
compression, and operational consequences. No historical evaluation was relabeled.

The dataset remains version 1.1.0 with hash
`70152d1ac15e827a0ff940cf5becf76701578d0a62d773b4ad6e3823c3fa27ca`.
The split and provenance identities are unchanged. The corrected evaluator hash is
`390e5ace4d06d71578759f92ebb11d3e8e871e6a11842dd48448531dd1fd3b0b`,
and protocol 1.2.0 has hash
`274bef860c138a98d6f870ade391210903834cf732d934153b17b0d3fed43ea7`.
The same 21 train tasks and four candidates are frozen in the corrected rerun
manifest, which remains awaiting human authorization. No corrected pilot, Astra
validation, development run, or final-test run was executed during this correction.
