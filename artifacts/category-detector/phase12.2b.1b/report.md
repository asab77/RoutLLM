# Phase 12.2B.1b Bounded Category-Boundary Correction

## Outcome

The single bounded correction round produced a large TRAIN grouped-CV
improvement but changed zero predictions on the reused DEV set. Production
integration is therefore not recommended, and detector optimization stops here.

## TRAIN-only forensics

The baseline confused extraction and structured JSON in both directions at the
family level. Direct-recovery requests commonly identified source material such
as contacts, logs, tables, rows, entries, or precedence sections. Structured
transformation requests instead expressed operations such as mapping,
conditional inclusion, deduplication, joining, sorting, grouping, or counting.
JSON syntax itself was not discriminative.

## Frozen correction

The correction runs only when the learned prediction is `extraction` or
`structured_json`:

1. An explicit transformation operation selects `structured_json`.
2. Otherwise an explicit direct-recovery source selects `extraction`.
3. Otherwise the learned prediction is unchanged.

No other predicted category can be changed. The frozen version is
`phase12.2b.1b-extraction-json-v1`.

## TRAIN grouped-CV comparison

| Metric | Before | After |
|---|---:|---:|
| Correct | 113/140 | 129/140 |
| Accuracy | 80.71% | 92.14% |
| Macro-F1 | 0.8035 | 0.9206 |
| Extraction precision | 50.00% | 86.36% |
| Extraction recall | 50.00% | 95.00% |
| Structured-JSON precision | 50.00% | 100.00% |
| Structured-JSON recall | 45.00% | 80.00% |

The correction changed 16 TRAIN predictions, corrected all 16, and introduced
zero errors. Precision and recall for the other five categories were unchanged.
The semantic signals were present on 23/140 requests; only 16 required a changed
prediction.

## REUSED DEV REGRESSION CHECK

DEV was previously observed and is not treated as fresh held-out data.

| Metric | Before | Revised |
|---|---:|---:|
| Correct | 36/42 | 36/42 |
| Accuracy | 85.71% | 85.71% |
| Macro-F1 | 0.8490 | 0.8490 |
| Extraction precision | 66.67% | 66.67% |
| Extraction recall | 33.33% | 33.33% |
| Structured-JSON precision | 55.56% | 55.56% |
| Structured-JSON recall | 83.33% | 83.33% |

All other category metrics were also unchanged. The frozen correction signals
matched two DEV requests but agreed with their existing learned predictions, so
the number of changed predictions was zero.

Confusion matrix, rows=true and columns=predicted in the order classification,
coding, extraction, structured_json, qa, reasoning, summarization:

```text
[[6, 0, 0, 0, 0, 0, 0],
 [0, 6, 0, 0, 0, 0, 0],
 [0, 0, 2, 4, 0, 0, 0],
 [0, 0, 1, 5, 0, 0, 0],
 [0, 0, 0, 0, 5, 1, 0],
 [0, 0, 0, 0, 0, 6, 0],
 [0, 0, 0, 0, 0, 0, 6]]
```

## Routing regression

Routing behavior was identical to Phase 12.2B.1:

- Same selected model: 40/42.
- Changed selected model: 2/42.
- Same fallback state: 41/42.
- Changed fallback state: 1/42.
- Projected-cost delta: +$0.00065775, or +8.82% on this set.

## Recommendation

Do not integrate the seven-way automatic detector into production. The smallest
alternative is to keep explicit category selection for users or workflows that
want adaptive routing, while uncategorized normal chat bypasses category-based
adaptive routing and uses one configured default model. The detector may remain
an offline/advisory experiment, but no further optimization round is justified
by this benchmark.

## Safety

Correction design used TRAIN prompts and TRAIN family-aware CV only. Individual
DEV prompt text was not inspected. DEV was run once after freeze solely as a
reused regression set. FINAL remained untouched. No provider, AWS, semantic
judge, browser, embedding, or external-download call occurred. The frozen
predictor remained SHA-256
`502db83a54c4072c9741a8e4c406498ad88ec97e03bce1ddaf3e0a1b0001c0aa`,
the routing threshold remained 0.80, and production API/frontend behavior was
not modified.
