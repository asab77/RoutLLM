# Phase 12.2B.1 Offline Category Detector Experiment

## Decision

The frozen detector is not yet sufficiently reliable for Phase 12.2B.2
production integration. DEV accuracy was 36/42 (85.71%) with macro-F1 0.8490,
but extraction recall was only 2/6 (33.33%) and structured-JSON precision was
5/9 (55.56%). The 42-request DEV set is too small and too benchmark-shaped to
establish ordinary-chat generalization.

If an ambiguity policy must be chosen from the evaluated options, always accept
the classifier prediction is the least disruptive. The margin-0.20 policy was
perfect on accepted DEV predictions but abstained on 21/42 requests, which would
make normal chat require manual categorization too often. There is no evidence
supporting a silent fallback category.

## Data identity

- TRAIN: 140 unique prompt requests, 20 per category.
- DEV: 42 unique prompt requests, 6 per category.
- FINAL: not accessed.
- TRAIN prompt projection SHA-256: `332816aee6edcae3f792082768aae200df8113df6fccafc1979d3f7bb231de4e`
- DEV prompt projection SHA-256: `acbaa8c682993834dbbddbefac256ba9cf01bf5146532156f705dc77db43e22c`

## Frozen configuration

- Normalization: Unicode-preserving whitespace collapse plus case-folding.
- Word TF-IDF: 1–2 grams, minimum document frequency 1, sublinear term frequency.
- Character TF-IDF: `char_wb` 3–5 grams, minimum document frequency 1, sublinear
  term frequency, maximum 30,000 features.
- Logistic regression: C=1.0, `lbfgs`, 3,000 maximum iterations, random state
  20260929.
- CV: five-fold shuffled `StratifiedGroupKFold`, grouped by task family.
- Hybrid: use a deterministic rule when one fires; otherwise use logistic
  regression.
- Frozen primary detector: standalone learned classifier. The high-precision
  rules did not change its TRAIN CV predictions.
- Abstention candidates: top score >=0.50; margin >=0.20; top score >=0.50 and
  margin >=0.15. Scores are not treated as calibrated confidence.

## TRAIN/CV results

Rules covered 44/140 (31.43%), correctly classified all 44 covered requests
(100% precision), and abstained on 96/140 (68.57%). Coverage was coding 20/20,
extraction 4/20, summarization 20/20, and zero for the other four categories.

Both the learned classifier and hybrid produced 113/140 correct predictions
(80.71%). Learned macro-F1 was 0.8035; hybrid macro-F1 was also 0.8035. The C=4
comparison also produced 113/140 but a slightly lower macro-F1 of 0.8024.

TRAIN learned/hybrid per-category precision and recall:

| Category | Precision | Recall |
|---|---:|---:|
| classification | 90.00% | 90.00% |
| coding | 100.00% | 100.00% |
| extraction | 50.00% | 50.00% |
| structured_json | 50.00% | 45.00% |
| qa | 86.96% | 100.00% |
| reasoning | 84.21% | 80.00% |
| summarization | 100.00% | 100.00% |

TRAIN abstention results:

| Policy | Accepted | Abstained | Accepted accuracy | Accepted errors |
|---|---:|---:|---:|---:|
| top >=0.50 | 58/140 | 82/140 | 100.00% | 0 |
| margin >=0.20 | 67/140 | 73/140 | 97.01% | 2 |
| top >=0.50 and margin >=0.15 | 58/140 | 82/140 | 100.00% | 0 |

## Single frozen DEV evaluation

Rules covered 12/42 (28.57%), correctly classified all 12 covered requests,
and left 30 unresolved. Coverage was six coding and six summarization requests.

The learned classifier and hybrid both produced 36/42 correct predictions
(85.71%) with macro-F1 0.8490.

| Category | Precision | Recall | Correct/actual |
|---|---:|---:|---:|
| classification | 100.00% | 100.00% | 6/6 |
| coding | 100.00% | 100.00% | 6/6 |
| extraction | 66.67% | 33.33% | 2/6 |
| structured_json | 55.56% | 83.33% | 5/6 |
| qa | 100.00% | 83.33% | 5/6 |
| reasoning | 85.71% | 100.00% | 6/6 |
| summarization | 100.00% | 100.00% | 6/6 |

Confusion matrix, with rows=true and columns=predicted in the order
classification, coding, extraction, structured_json, qa, reasoning,
summarization:

```text
[[6, 0, 0, 0, 0, 0, 0],
 [0, 6, 0, 0, 0, 0, 0],
 [0, 0, 2, 4, 0, 0, 0],
 [0, 0, 1, 5, 0, 0, 0],
 [0, 0, 0, 0, 5, 1, 0],
 [0, 0, 0, 0, 0, 6, 0],
 [0, 0, 0, 0, 0, 0, 6]]
```

DEV abstention results:

| Policy | Accepted | Abstained | Accepted accuracy | Accepted errors |
|---|---:|---:|---:|---:|
| top >=0.50 | 15/42 | 27/42 | 100.00% | 0 |
| margin >=0.20 | 21/42 | 21/42 | 100.00% | 0 |
| top >=0.50 and margin >=0.15 | 15/42 | 27/42 | 100.00% | 0 |

## Offline routing impact

Using the frozen production predictor and unchanged threshold 0.80:

- Correct detected categories: 36/42; incorrect: 6/42.
- Same selected model: 40/42; changed selected model: 2/42.
- Same fallback state: 41/42; changed fallback state: 1/42.
- True-category projected cost total: $0.00745730.
- Detected-category projected cost total: $0.00811505.
- Delta: +$0.00065775 (+8.82% for this small request set).

Route-changing errors:

| Task | Confusion | True route | Detected route | Fallback change | Cost delta |
|---|---|---|---|---|---:|
| extraction-medium-010 | extraction -> structured_json | candidate-gpt-6-luna | candidate-nemotron-3.5-lightning | no | -$0.00004545 |
| qa-hard-007 | qa -> reasoning | candidate-gpt-6-luna | candidate-gemini-3-flash | false -> true | +$0.00070320 |

The other four category errors did not change the selected model or fallback
state. Historical acceptability was not evaluated because provider outcomes and
validation labels were forbidden inputs.

## Limitations

- DEV contains only six requests per category.
- TRAIN/DEV families come from one controlled benchmark and are not a sample of
  ordinary chat traffic.
- Extraction versus structured JSON remains the dominant ambiguity.
- TF-IDF scores are not calibrated probabilities.
- The routing comparison omits the benchmark system prompt because it is not
  exposed by the safe projection; the true/detected comparison nevertheless
  holds every available feature except category constant.
- Projected costs are policy estimates, not realized provider costs.

## Safety

No provider, AWS, semantic-judge, browser, embedding, or external-download call
was made. The frozen predictor remained at SHA-256
`502db83a54c4072c9741a8e4c406498ad88ec97e03bce1ddaf3e0a1b0001c0aa`.
Production API behavior and the 0.80 routing threshold were not changed.
