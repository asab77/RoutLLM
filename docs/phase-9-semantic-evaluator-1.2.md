# Phase 9 semantic summarization evaluator 1.2

Evaluator 1.1 combined required-fact coverage, instruction compliance, and a single
`factual_consistency` score, then forced quality to zero whenever factual consistency
was below 1. A single opaque model judgment could therefore override complete coverage
and correct formatting without identifying the semantic defect that caused the failure.

Evaluator 1.2 replaces that field with required-fact coverage, instruction compliance,
and six concrete material-error lists: unsupported claims, contradictions, unsupported
causal claims, attribution errors, quantity/time errors, and certainty distortions. The
judge supplies concise evidence descriptions only. It is not asked for chain of thought.

A material unsupported claim adds a substantive proposition the source does not support,
including a new event, actor, quantity, time, causal relation, motive, diagnosis,
attribution, certainty level, or a contradiction. Ordinary lexical paraphrase, semantic
compression, equivalent quantities, equivalent certainty, and descriptive wording are
allowed when they do not add such a proposition. Temporal order by itself never licenses
a causal claim.

Aggregation is deterministic. Deterministic format and forbidden-claim checks must pass.
Any nonempty material-error list yields zero quality. Otherwise quality is the lower of
required-fact coverage and instruction compliance, and the existing task threshold applies.
This prevents a high score on one dimension from compensating for missing required facts.

The local adversarial suite contains 24 balanced supported and unsupported cases across
the required boundary phenomena. The planned live validation contains exactly eight blind
cases, four supported and four unsupported. Expected labels are stored separately from the
judge requests. Evaluator 1.2 is eligible for benchmark use only if all eight calls return
valid structured output and all eight classifications match their hidden expectations.
Any mismatch stops the validation for human review.

Offline replay mapped the stored Foundation V2 and V3 summarization judgments into the new
deterministic aggregation without changing their artifacts. No label changed in either
32-attempt run. In the historical Routing Benchmark pilot, two accepted easy-summary rows
would become failures because their recorded required-fact coverage was 2/3, below the
unchanged 0.8 threshold. The medium Sonnet row remains a failure: its historical positive
control was confounded because it combined reasonable descriptive wording with a causal
claim based only on sequence. Evaluator 1.2 treats the descriptive wording as supported
when isolated and still rejects the unsupported causal relation.

The eight-call Astra validation is estimated at $0.11663 using 4,463 approximate input
tokens and 1,440 expected output tokens. The conservative 256-output-token allowance is
$0.14703. The corrected 21-task pilot now estimates $0.15872 of judge cost and $0.20993709
total including candidates and the existing 10% contingency.
