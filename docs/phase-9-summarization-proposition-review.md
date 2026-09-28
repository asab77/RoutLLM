# Phase 9 summarization proposition specification — human review

- Specification: 1.0.0
- Proposed Routing Benchmark dataset: 1.2.0
- Proposed dataset SHA-256: `1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005`
- Implemented evaluator: unchanged at historical 1.2.0
- Status: human review required; no paid validation authorized

## Coverage rule

Coverage is the exact sum of ENTAILED proposition weights divided by total weight. Any AMBIGUOUS verdict makes the evaluation incomplete. Split weights preserve each legacy requirement's total weight. The existing 0.8 threshold is recommended for human approval because it represents at least 80% of benchmark-owned semantic weight; it is not selected or implemented by this specification.

## Material-error contract

Every finding requires the candidate claim, specific source evidence, one frozen error category, materiality (`material` or `borderline`), and a concise reason. Categories: UNSUPPORTED_MATERIAL_CLAIM, CONTRADICTION, UNSUPPORTED_CAUSATION, ATTRIBUTION_ERROR, QUANTITY_OR_TIME_ERROR, CERTAINTY_DISTORTION.

Clearly supported characterizations are not errors. Clearly unsupported substantive assertions are material errors. Genuinely borderline interpretive wording produces AMBIGUOUS and requires review; it is not an automatic veto.

## rb12-summarization-easy-001

**Applies to:** routing-v1.2:summarization-easy-001

**Source:** On Monday the team opened the migration. Tuesday testing found a timezone bug. Wednesday the bug was fixed, and Thursday the migration completed. A separate office lunch occurred Friday.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-easy-001-r01` — Accurately state that testing found a timezone bug Tuesday. | 1/1 | Tuesday testing found a timezone bug. | Tuesday | — |
| `rb12-summarization-easy-001-r02` — Accurately state that the bug was fixed Wednesday. | 1/1 | Wednesday the bug was fixed, and Thursday the migration completed. | Wednesday | — |
| `rb12-summarization-easy-001-r03` — Accurately state that the migration completed Thursday. | 1/1 | Wednesday the bug was fixed, and Thursday the migration completed. | Thursday | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-easy-001-r01` | 1 | 2/3 |
| `rb12-summarization-easy-001-r02` | 1 | 2/3 |
| `rb12-summarization-easy-001-r03` | 1 | 2/3 |
- Material traps: the migration failed

## rb12-summarization-easy-002

**Applies to:** routing-v1.2:summarization-easy-002

**Source:** The permit request arrived Friday. Staff requested a missing diagram Monday, received it Wednesday, and approved the permit Thursday. A billing address changed Tuesday but did not affect review.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-easy-002-r01` — Accurately state that staff requested a missing diagram Monday. | 1/1 | Staff requested a missing diagram Monday, received it Wednesday, and approved the permit Thursday. | Monday | — |
| `rb12-summarization-easy-002-r02` — Accurately state that the diagram arrived Wednesday. | 1/1 | Staff requested a missing diagram Monday, received it Wednesday, and approved the permit Thursday. | Wednesday | — |
| `rb12-summarization-easy-002-r03` — Accurately state that the permit was approved Thursday. | 1/1 | Staff requested a missing diagram Monday, received it Wednesday, and approved the permit Thursday. | Thursday | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-easy-002-r01` | 1 | 2/3 |
| `rb12-summarization-easy-002-r02` | 1 | 2/3 |
| `rb12-summarization-easy-002-r03` | 1 | 2/3 |
- Material traps: the billing change delayed approval

## rb12-summarization-easy-003

**Applies to:** routing-v1.2:summarization-easy-003

**Source:** Maya drafted the policy, Rafi reviewed its legal terms, and Chen approved the final version. Inez attended the meeting but had no approval role.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-easy-003-r01` — Accurately state that Maya drafted the policy. | 1/1 | Maya drafted the policy, Rafi reviewed its legal terms, and Chen approved the final version. | May, Maya | — |
| `rb12-summarization-easy-003-r02` — Accurately state that Rafi reviewed the legal terms. | 1/1 | Maya drafted the policy, Rafi reviewed its legal terms, and Chen approved the final version. | Rafi | — |
| `rb12-summarization-easy-003-r03` — Accurately state that Chen approved the final version. | 1/1 | Maya drafted the policy, Rafi reviewed its legal terms, and Chen approved the final version. | Chen | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-easy-003-r01` | 1 | 2/3 |
| `rb12-summarization-easy-003-r02` | 1 | 2/3 |
| `rb12-summarization-easy-003-r03` | 1 | 2/3 |
- Material traps: Inez approved the policy

## rb12-summarization-easy-004

**Applies to:** routing-v1.2:summarization-easy-004

**Source:** Jordan collected customer feedback, Priya converted it into requirements, and Luis authorized the release. Morgan observed the review but cast no vote.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-easy-004-r01` — Accurately state that Jordan collected customer feedback. | 1/1 | Jordan collected customer feedback, Priya converted it into requirements, and Luis authorized the release. | Jordan | — |
| `rb12-summarization-easy-004-r02` — Accurately state that Priya wrote the requirements. | 1/1 | Jordan collected customer feedback, Priya converted it into requirements, and Luis authorized the release. | Priya | — |
| `rb12-summarization-easy-004-r03` — Accurately state that Luis authorized the release. | 1/1 | Jordan collected customer feedback, Priya converted it into requirements, and Luis authorized the release. | Luis | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-easy-004-r01` | 1 | 2/3 |
| `rb12-summarization-easy-004-r02` | 1 | 2/3 |
| `rb12-summarization-easy-004-r03` | 1 | 2/3 |
- Material traps: Morgan authorized the release

## rb12-summarization-easy-005

**Applies to:** routing-v1.2:summarization-easy-005

**Source:** At 09:10 an expired certificate blocked logins. Operations renewed it at 09:32, and access recovered by 09:36. No customer data was lost.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-easy-005-r01` — Accurately state that an expired certificate blocked logins. | 1/1 | At 09:10 an expired certificate blocked logins. | — | — |
| `rb12-summarization-easy-005-r02` — Accurately state that operations renewed the certificate at 09:32. | 1/1 | Operations renewed it at 09:32, and access recovered by 09:36. | 09:32, 09, 32 | — |
| `rb12-summarization-easy-005-r03` — Accurately state that access recovered by 09:36. | 1/1 | Operations renewed it at 09:32, and access recovered by 09:36. | 09:36, 09, 36 | — |
| `rb12-summarization-easy-005-r04` — Accurately state that no customer data was lost. | 1/1 | No customer data was lost. | polarity/constraint | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-easy-005-r01` | 1 | 3/4 |
| `rb12-summarization-easy-005-r02` | 1 | 3/4 |
| `rb12-summarization-easy-005-r03` | 1 | 3/4 |
| `rb12-summarization-easy-005-r04` | 1 | 3/4 |
- Material traps: customer data was lost

## rb12-summarization-medium-001

**Applies to:** routing-v1.2:summarization-medium-001

**Source:** A routing rule deployed at 14:05 sent checkout traffic to an unavailable service. The rule was rolled back at 14:17; checkout recovered at 14:20, and no payments were duplicated.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-001-r01` — Accurately state that a routing rule sent checkout traffic to an unavailable service. | 1/1 | A routing rule deployed at 14:05 sent checkout traffic to an unavailable service. | — | — |
| `rb12-summarization-medium-001-r02` — Accurately state that the rule was rolled back at 14:17. | 1/1 | The rule was rolled back at 14:17; checkout recovered at 14:20, and no payments were duplicated. | 14:17, 14, 17 | — |
| `rb12-summarization-medium-001-r03` — Accurately state that checkout recovered at 14:20. | 1/1 | The rule was rolled back at 14:17; checkout recovered at 14:20, and no payments were duplicated. | 14:20, 14, 20 | — |
| `rb12-summarization-medium-001-r04` — Accurately state that no payments were duplicated. | 1/1 | The rule was rolled back at 14:17; checkout recovered at 14:20, and no payments were duplicated. | polarity/constraint | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-001-r01` | 1 | 3/4 |
| `rb12-summarization-medium-001-r02` | 1 | 3/4 |
| `rb12-summarization-medium-001-r03` | 1 | 3/4 |
| `rb12-summarization-medium-001-r04` | 1 | 3/4 |
- Material traps: payments were duplicated

## rb12-summarization-medium-002

**Applies to:** routing-v1.2:summarization-medium-002

**Source:** The campaign reached 12,400 people, generated 620 visits, and produced 31 purchases. Spend was $1,550. The design team also tested three unused logos.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-002-r01` — Accurately state that 12,400 people were reached. | 1/1 | The campaign reached 12,400 people, generated 620 visits, and produced 31 purchases. | 12,400 people | ALLOWED: 12,400 people = 12.4 thousand people |
| `rb12-summarization-medium-002-r02` — Accurately state that 620 visits were generated. | 1/1 | The campaign reached 12,400 people, generated 620 visits, and produced 31 purchases. | 620 | — |
| `rb12-summarization-medium-002-r03` — Accurately state that 31 purchases resulted. | 1/1 | The campaign reached 12,400 people, generated 620 visits, and produced 31 purchases. | 31 | — |
| `rb12-summarization-medium-002-r04` — Accurately state that spend was $1,550. | 1/1 | Spend was $1,550. | $1,550, 1,550 | ALLOWED: $1,550 = $1.55 thousand |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-002-r01` | 1 | 3/4 |
| `rb12-summarization-medium-002-r02` | 1 | 3/4 |
| `rb12-summarization-medium-002-r03` | 1 | 3/4 |
| `rb12-summarization-medium-002-r04` | 1 | 3/4 |
- Material traps: the campaign made 620 purchases

## rb12-summarization-medium-003

**Applies to:** routing-v1.2:summarization-medium-003

**Source:** The workshop registered 480 people; 360 attended, 288 completed the lab, and 250 submitted feedback. Catering prepared 400 lunches.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-003-r01` — Accurately state that 480 people registered. | 1/1 | The workshop registered 480 people; 360 attended, 288 completed the lab, and 250 submitted feedback. | 480 people | — |
| `rb12-summarization-medium-003-r02` — Accurately state that 360 attended. | 1/1 | The workshop registered 480 people; 360 attended, 288 completed the lab, and 250 submitted feedback. | 360 | CONDITIONAL: 360 of 480 = 75% attendance |
| `rb12-summarization-medium-003-r03` — Accurately state that 288 completed the lab. | 1/1 | The workshop registered 480 people; 360 attended, 288 completed the lab, and 250 submitted feedback. | 288 | — |
| `rb12-summarization-medium-003-r04` — Accurately state that 250 submitted feedback. | 1/1 | The workshop registered 480 people; 360 attended, 288 completed the lab, and 250 submitted feedback. | 250 | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-003-r01` | 1 | 3/4 |
| `rb12-summarization-medium-003-r02` | 1 | 3/4 |
| `rb12-summarization-medium-003-r03` | 1 | 3/4 |
| `rb12-summarization-medium-003-r04` | 1 | 3/4 |
- Material traps: 400 people attended

## rb12-summarization-medium-004

**Applies to:** routing-v1.2:summarization-medium-004

**Source:** Employees may work remotely two days per week. New hires must work onsite during their first month, while documented accessibility accommodations can override that restriction.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-004-r01` — Accurately state that remote work is allowed two days per week. | 1/1 | Employees may work remotely two days per week. | — | — |
| `rb12-summarization-medium-004-r02` — Accurately state that new hires must be onsite for their first month. | 1/1 | New hires must work onsite during their first month, while documented accessibility accommodations can override that restriction. | polarity/constraint | — |
| `rb12-summarization-medium-004-r03` — Accurately state that accessibility accommodations can override the restriction. | 1/1 | New hires must work onsite during their first month, while documented accessibility accommodations can override that restriction. | — | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-004-r01` | 1 | 2/3 |
| `rb12-summarization-medium-004-r02` | 1 | 2/3 |
| `rb12-summarization-medium-004-r03` | 1 | 2/3 |
- Material traps: new hires may always work remotely

## rb12-summarization-medium-005

**Applies to:** routing-v1.2:summarization-medium-005

**Source:** Expense reports are due within 20 days. Employees on approved leave receive five extra days, but cash advances must always be reconciled within 10 days.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-005-r01` — Accurately state that expense reports are due within 20 days. | 1/1 | Expense reports are due within 20 days. | 20 days | — |
| `rb12-summarization-medium-005-r02` — Accurately state that approved leave adds five days. | 1/1 | Employees on approved leave receive five extra days, but cash advances must always be reconciled within 10 days. | — | CONDITIONAL: 20 days + 5 days = a 25-day leave deadline |
| `rb12-summarization-medium-005-r03` — Accurately state that cash advances remain due within 10 days. | 1/1 | Employees on approved leave receive five extra days, but cash advances must always be reconciled within 10 days. | 10 days | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-005-r01` | 1 | 2/3 |
| `rb12-summarization-medium-005-r02` | 1 | 2/3 |
| `rb12-summarization-medium-005-r03` | 1 | 2/3 |
- Material traps: leave extends the cash-advance deadline

## rb12-summarization-easy-006

**Applies to:** routing-v1.2:summarization-easy-006

**Source:** The original launch date was May 4. Supplier delays moved it to May 18. After expedited shipping, the final approved date became May 12.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-easy-006-r01` — Accurately state that the original date was May 4. | 1/1 | The original launch date was May 4. | May 4, 4 | — |
| `rb12-summarization-easy-006-r02` — Accurately state that supplier delays moved it to May 18. | 1/1 | Supplier delays moved it to May 18. | May 18, 18 | — |
| `rb12-summarization-easy-006-r03` — Accurately state that the final approved date is May 12. | 1/1 | After expedited shipping, the final approved date became May 12. | May 12, 12 | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-easy-006-r01` | 1 | 2/3 |
| `rb12-summarization-easy-006-r02` | 1 | 2/3 |
| `rb12-summarization-easy-006-r03` | 1 | 2/3 |
- Material traps: the final date is May 18

## rb12-summarization-medium-006

**Applies to:** routing-v1.2:summarization-medium-006

**Source:** The storage target began at 80 TB, increased to 110 TB after acquisition, and was reduced to 95 TB after archival. The approved capacity plan now uses 95 TB.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-006-r01` — Accurately state that the target began at 80 TB. | 1/1 | The storage target began at 80 TB, increased to 110 TB after acquisition, and was reduced to 95 TB after archival. | 80 TB | — |
| `rb12-summarization-medium-006-r02` — Accurately state that acquisition raised it to 110 TB. | 1/1 | The storage target began at 80 TB, increased to 110 TB after acquisition, and was reduced to 95 TB after archival. | 110 TB | CONDITIONAL: 110 TB is a 30 TB increase from 80 TB |
| `rb12-summarization-medium-006-r03` — Accurately state that archival reduced the approved target to 95 TB. | 1/1 | The storage target began at 80 TB, increased to 110 TB after acquisition, and was reduced to 95 TB after archival. | 95 TB | CONDITIONAL: 95 TB is a 15 TB reduction from 110 TB |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-006-r01` | 1 | 2/3 |
| `rb12-summarization-medium-006-r02` | 1 | 2/3 |
| `rb12-summarization-medium-006-r03` | 1 | 2/3 |
- Material traps: the approved target is 110 TB

## rb12-summarization-easy-007

**Applies to:** routing-v1.2:summarization-easy-007

**Source:** A cooling fan failed, causing the server to overheat and shut down. Replacing the fan restored normal temperature; the database required no repair.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-easy-007-r01a` — A cooling fan failed. | 1/2 | A cooling fan failed, causing the server to overheat and shut down. | — | — |
| `rb12-summarization-easy-007-r01b` — The fan failure caused the server to shut down. | 1/2 | A cooling fan failed, causing the server to overheat and shut down. | — | — |
| `rb12-summarization-easy-007-r02` — Accurately state that replacing the fan restored normal temperature. | 1/1 | Replacing the fan restored normal temperature; the database required no repair. | — | — |
| `rb12-summarization-easy-007-r03` — Accurately state that the database required no repair. | 1/1 | Replacing the fan restored normal temperature; the database required no repair. | polarity/constraint | — |

**Total weight:** 3; **minimum meaningful omission:** 1/2; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-easy-007-r01a` | 1/2 | 5/6 |
| `rb12-summarization-easy-007-r01b` | 1/2 | 5/6 |
| `rb12-summarization-easy-007-r02` | 1 | 2/3 |
| `rb12-summarization-easy-007-r03` | 1 | 2/3 |
- Composite review: **SPLIT** — Accurately state that a cooling fan failed and the server shut down. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Material traps: the database caused the shutdown

## rb12-summarization-easy-008

**Applies to:** routing-v1.2:summarization-easy-008

**Source:** A malformed catalog entry caused the importer to reject the nightly batch. Correcting the entry allowed the rerun to finish; the importer code was unchanged.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-easy-008-r01` — Accurately state that a malformed catalog entry caused the batch rejection. | 1/1 | A malformed catalog entry caused the importer to reject the nightly batch. | — | — |
| `rb12-summarization-easy-008-r02` — Accurately state that correcting the entry allowed the rerun to finish. | 1/1 | Correcting the entry allowed the rerun to finish; the importer code was unchanged. | — | — |
| `rb12-summarization-easy-008-r03` — Accurately state that the importer code was unchanged. | 1/1 | Correcting the entry allowed the rerun to finish; the importer code was unchanged. | polarity/constraint | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-easy-008-r01` | 1 | 2/3 |
| `rb12-summarization-easy-008-r02` | 1 | 2/3 |
| `rb12-summarization-easy-008-r03` | 1 | 2/3 |
- Material traps: an importer code defect caused the rejection

## rb12-summarization-hard-001

**Applies to:** routing-v1.2:summarization-hard-001

**Source:** Asha proposed retaining the vendor. Bo favored a new bid. Cy abstained because of a conflict. The committee voted 4–2 to seek a new vendor.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-001-r01` — Accurately state that Asha favored retaining the vendor. | 1/1 | Asha proposed retaining the vendor. | Asha | — |
| `rb12-summarization-hard-001-r02` — Accurately state that Bo favored a new bid. | 1/1 | Bo favored a new bid. | Bo | — |
| `rb12-summarization-hard-001-r03` — Accurately state that Cy abstained due to a conflict. | 1/1 | Cy abstained because of a conflict. | Cy | — |
| `rb12-summarization-hard-001-r04` — Accurately state that the committee voted 4–2 for a new vendor search. | 1/1 | The committee voted 4–2 to seek a new vendor. | 4, 2 | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-001-r01` | 1 | 3/4 |
| `rb12-summarization-hard-001-r02` | 1 | 3/4 |
| `rb12-summarization-hard-001-r03` | 1 | 3/4 |
| `rb12-summarization-hard-001-r04` | 1 | 3/4 |
- Material traps: the vote retained the vendor

## rb12-summarization-hard-002

**Applies to:** routing-v1.2:summarization-hard-002

**Source:** Nora recommended extending the trial, Omar preferred purchase, and Pia requested more security evidence. The board postponed purchase 5–1 pending the security review.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-002-r01` — Accurately state that Nora recommended extending the trial. | 1/1 | Nora recommended extending the trial, Omar preferred purchase, and Pia requested more security evidence. | Nora | — |
| `rb12-summarization-hard-002-r02` — Accurately state that Omar preferred purchase. | 1/1 | Nora recommended extending the trial, Omar preferred purchase, and Pia requested more security evidence. | Omar | — |
| `rb12-summarization-hard-002-r03` — Accurately state that Pia requested security evidence. | 1/1 | Nora recommended extending the trial, Omar preferred purchase, and Pia requested more security evidence. | Pia | — |
| `rb12-summarization-hard-002-r04` — Accurately state that the board voted 5–1 to postpone purchase. | 1/1 | The board postponed purchase 5–1 pending the security review. | 5, 1 | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-002-r01` | 1 | 3/4 |
| `rb12-summarization-hard-002-r02` | 1 | 3/4 |
| `rb12-summarization-hard-002-r03` | 1 | 3/4 |
| `rb12-summarization-hard-002-r04` | 1 | 3/4 |
- Material traps: the board approved purchase

## rb12-summarization-medium-007

**Applies to:** routing-v1.2:summarization-medium-007

**Source:** To publish a report, an analyst uploads the draft, a reviewer resolves factual issues, and an editor approves formatting. Publication occurs only after all three stages pass.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-007-r01` — Accurately state that the analyst uploads the draft. | 1/1 | To publish a report, an analyst uploads the draft, a reviewer resolves factual issues, and an editor approves formatting. | — | — |
| `rb12-summarization-medium-007-r02` — Accurately state that a reviewer resolves factual issues. | 1/1 | To publish a report, an analyst uploads the draft, a reviewer resolves factual issues, and an editor approves formatting. | — | — |
| `rb12-summarization-medium-007-r03` — Accurately state that an editor approves formatting. | 1/1 | To publish a report, an analyst uploads the draft, a reviewer resolves factual issues, and an editor approves formatting. | — | — |
| `rb12-summarization-medium-007-r04` — Accurately state that publication requires all stages to pass. | 1/1 | Publication occurs only after all three stages pass. | polarity/constraint | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-007-r01` | 1 | 3/4 |
| `rb12-summarization-medium-007-r02` | 1 | 3/4 |
| `rb12-summarization-medium-007-r03` | 1 | 3/4 |
| `rb12-summarization-medium-007-r04` | 1 | 3/4 |
- Material traps: publication occurs before review

## rb12-summarization-hard-003

**Applies to:** routing-v1.2:summarization-hard-003

**Source:** A refund request is first matched to the original payment, then checked for policy eligibility, and finally approved by Finance. Funds are released only after all three checks succeed.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-003-r01` — Accurately state that the request is matched to the original payment. | 1/1 | A refund request is first matched to the original payment, then checked for policy eligibility, and finally approved by Finance. | — | — |
| `rb12-summarization-hard-003-r02` — Accurately state that policy eligibility is checked. | 1/1 | A refund request is first matched to the original payment, then checked for policy eligibility, and finally approved by Finance. | — | — |
| `rb12-summarization-hard-003-r03` — Accurately state that Finance gives final approval. | 1/1 | A refund request is first matched to the original payment, then checked for policy eligibility, and finally approved by Finance. | Finance | — |
| `rb12-summarization-hard-003-r04` — Accurately state that funds require all three checks. | 1/1 | Funds are released only after all three checks succeed. | polarity/constraint | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-003-r01` | 1 | 3/4 |
| `rb12-summarization-hard-003-r02` | 1 | 3/4 |
| `rb12-summarization-hard-003-r03` | 1 | 3/4 |
| `rb12-summarization-hard-003-r04` | 1 | 3/4 |
- Material traps: funds are released before Finance approval

## rb12-summarization-medium-008

**Applies to:** routing-v1.2:summarization-medium-008

**Source:** Plan North costs $18 monthly and includes 20 GB. Plan South costs $24 and includes 50 GB plus roaming. Both include phone support.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-008-r01a` — Plan North costs $18 monthly. | 1/2 | Plan North costs $18 monthly and includes 20 GB. | $18, 18, Plan North | — |
| `rb12-summarization-medium-008-r01b` — Plan North includes 20 GB. | 1/2 | Plan North costs $18 monthly and includes 20 GB. | 20 GB, Plan North | — |
| `rb12-summarization-medium-008-r02a` — Plan South costs $24 monthly. | 1/3 | Plan South costs $24 and includes 50 GB plus roaming. | $24, 24, Plan South | — |
| `rb12-summarization-medium-008-r02b` — Plan South includes 50 GB. | 1/3 | Plan South costs $24 and includes 50 GB plus roaming. | 50 GB, Plan South | — |
| `rb12-summarization-medium-008-r02c` — Plan South includes roaming. | 1/3 | Plan South costs $24 and includes 50 GB plus roaming. | Plan South | — |
| `rb12-summarization-medium-008-r03` — Accurately state that both include phone support. | 1/1 | Both include phone support. | — | — |

**Total weight:** 3; **minimum meaningful omission:** 1/3; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-008-r01a` | 1/2 | 5/6 |
| `rb12-summarization-medium-008-r01b` | 1/2 | 5/6 |
| `rb12-summarization-medium-008-r02a` | 1/3 | 8/9 |
| `rb12-summarization-medium-008-r02b` | 1/3 | 8/9 |
| `rb12-summarization-medium-008-r02c` | 1/3 | 8/9 |
| `rb12-summarization-medium-008-r03` | 1 | 2/3 |
- Composite review: **SPLIT** — Accurately state that North costs $18 and includes 20 GB. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Composite review: **SPLIT** — Accurately state that South costs $24 and includes 50 GB plus roaming. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Material traps: North includes more data than South

## rb12-summarization-medium-009

**Applies to:** routing-v1.2:summarization-medium-009

**Source:** Vendor Pine charges $900 setup plus $80 monthly and offers weekday support. Vendor Lake has no setup fee, charges $125 monthly, and includes 24/7 support.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-009-r01a` — Vendor Pine charges a $900 setup fee. | 1/2 | Vendor Pine charges $900 setup plus $80 monthly and offers weekday support. | $900, 900, Vendor Pine | — |
| `rb12-summarization-medium-009-r01b` — Vendor Pine charges $80 monthly. | 1/2 | Vendor Pine charges $900 setup plus $80 monthly and offers weekday support. | $80, 80, Vendor Pine | — |
| `rb12-summarization-medium-009-r02` — Accurately state that Pine offers weekday support. | 1/1 | Vendor Pine charges $900 setup plus $80 monthly and offers weekday support. | — | — |
| `rb12-summarization-medium-009-r03a` — Vendor Lake has no setup fee. | 1/2 | Vendor Lake has no setup fee, charges $125 monthly, and includes 24/7 support. | polarity/constraint, Vendor Lake | — |
| `rb12-summarization-medium-009-r03b` — Vendor Lake charges $125 monthly. | 1/2 | Vendor Lake has no setup fee, charges $125 monthly, and includes 24/7 support. | $125, 125, Vendor Lake | — |
| `rb12-summarization-medium-009-r04` — Accurately state that Lake includes 24/7 support. | 1/1 | Vendor Lake has no setup fee, charges $125 monthly, and includes 24/7 support. | 24, 7 | — |

**Total weight:** 4; **minimum meaningful omission:** 1/2; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-009-r01a` | 1/2 | 7/8 |
| `rb12-summarization-medium-009-r01b` | 1/2 | 7/8 |
| `rb12-summarization-medium-009-r02` | 1 | 3/4 |
| `rb12-summarization-medium-009-r03a` | 1/2 | 7/8 |
| `rb12-summarization-medium-009-r03b` | 1/2 | 7/8 |
| `rb12-summarization-medium-009-r04` | 1 | 3/4 |
- Composite review: **SPLIT** — Accurately state that Pine charges $900 setup and $80 monthly. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Composite review: **SPLIT** — Accurately state that Lake has no setup fee and costs $125 monthly. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Material traps: Pine includes 24/7 support

## rb12-summarization-hard-004

**Applies to:** routing-v1.2:summarization-hard-004

**Source:** The preliminary survey suggests demand may rise by 8–12%, but the sample is small and the estimate is not a forecast. A larger survey begins next month.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-004-r01` — Accurately state that preliminary demand may rise 8–12%. | 1/1 | The preliminary survey suggests demand may rise by 8–12%, but the sample is small and the estimate is not a forecast. | 12%, may, 8, 12 | — |
| `rb12-summarization-hard-004-r02` — Accurately state that the sample is small. | 1/1 | The preliminary survey suggests demand may rise by 8–12%, but the sample is small and the estimate is not a forecast. | — | — |
| `rb12-summarization-hard-004-r03` — Accurately state that the estimate is not a forecast. | 1/1 | The preliminary survey suggests demand may rise by 8–12%, but the sample is small and the estimate is not a forecast. | polarity/constraint | — |
| `rb12-summarization-hard-004-r04` — Accurately state that a larger survey begins next month. | 1/1 | A larger survey begins next month. | — | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-004-r01` | 1 | 3/4 |
| `rb12-summarization-hard-004-r02` | 1 | 3/4 |
| `rb12-summarization-hard-004-r03` | 1 | 3/4 |
| `rb12-summarization-hard-004-r04` | 1 | 3/4 |
- Material traps: demand will definitely rise 12%

## rb12-summarization-hard-005

**Applies to:** routing-v1.2:summarization-hard-005

**Source:** An early sensor analysis indicates energy use could fall 4–7%, but winter data is missing and the result has not been independently replicated. A full-year study ends in December.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-005-r01` — Accurately state that early analysis suggests a 4–7% reduction. | 1/1 | An early sensor analysis indicates energy use could fall 4–7%, but winter data is missing and the result has not been independently replicated. | 7%, 4, 7 | — |
| `rb12-summarization-hard-005-r02` — Accurately state that winter data is missing. | 1/1 | An early sensor analysis indicates energy use could fall 4–7%, but winter data is missing and the result has not been independently replicated. | — | — |
| `rb12-summarization-hard-005-r03` — Accurately state that the result is not independently replicated. | 1/1 | An early sensor analysis indicates energy use could fall 4–7%, but winter data is missing and the result has not been independently replicated. | polarity/constraint | — |
| `rb12-summarization-hard-005-r04` — Accurately state that the full-year study ends in December. | 1/1 | A full-year study ends in December. | December | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-005-r01` | 1 | 3/4 |
| `rb12-summarization-hard-005-r02` | 1 | 3/4 |
| `rb12-summarization-hard-005-r03` | 1 | 3/4 |
| `rb12-summarization-hard-005-r04` | 1 | 3/4 |
- Material traps: energy use will certainly fall 7%

## rb12-summarization-hard-006

**Applies to:** routing-v1.2:summarization-hard-006

**Source:** Standard refunds require a receipt within 30 days. Gifts may use an order number instead. Clearance items are never refundable unless defective.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-006-r01` — Accurately state that standard refunds require a receipt within 30 days. | 1/1 | Standard refunds require a receipt within 30 days. | 30 days, polarity/constraint | — |
| `rb12-summarization-hard-006-r02` — Accurately state that gifts may use an order number. | 1/1 | Gifts may use an order number instead. | may | — |
| `rb12-summarization-hard-006-r03` — Accurately state that clearance items require a defect to be refundable. | 1/1 | Clearance items are never refundable unless defective. | polarity/constraint | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-006-r01` | 1 | 2/3 |
| `rb12-summarization-hard-006-r02` | 1 | 2/3 |
| `rb12-summarization-hard-006-r03` | 1 | 2/3 |
- Material traps: all clearance items are refundable

## rb12-summarization-hard-007

**Applies to:** routing-v1.2:summarization-hard-007

**Source:** Reservations can be changed without charge until noon the prior day. Flexible fares may change until departure, while group bookings always require coordinator approval.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-007-r01` — Accurately state that standard reservations are free to change until noon the prior day. | 1/1 | Reservations can be changed without charge until noon the prior day. | noon | — |
| `rb12-summarization-hard-007-r02` — Accurately state that flexible fares may change until departure. | 1/1 | Flexible fares may change until departure, while group bookings always require coordinator approval. | may | — |
| `rb12-summarization-hard-007-r03` — Accurately state that group bookings require coordinator approval. | 1/1 | Flexible fares may change until departure, while group bookings always require coordinator approval. | polarity/constraint | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-007-r01` | 1 | 2/3 |
| `rb12-summarization-hard-007-r02` | 1 | 2/3 |
| `rb12-summarization-hard-007-r03` | 1 | 2/3 |
- Material traps: group bookings never require approval

## rb12-summarization-medium-010

**Applies to:** routing-v1.2:summarization-medium-010

**Source:** The bridge design passed safety review in January. Funding was approved in March. Construction began in June and is scheduled to finish in November.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-010-r01` — Accurately state that safety review passed in January. | 1/1 | The bridge design passed safety review in January. | January | — |
| `rb12-summarization-medium-010-r02` — Accurately state that funding was approved in March. | 1/1 | Funding was approved in March. | March | — |
| `rb12-summarization-medium-010-r03` — Accurately state that construction began in June. | 1/1 | Construction began in June and is scheduled to finish in November. | June | — |
| `rb12-summarization-medium-010-r04` — Accurately state that completion is scheduled for November. | 1/1 | Construction began in June and is scheduled to finish in November. | November | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-010-r01` | 1 | 3/4 |
| `rb12-summarization-medium-010-r02` | 1 | 3/4 |
| `rb12-summarization-medium-010-r03` | 1 | 3/4 |
| `rb12-summarization-medium-010-r04` | 1 | 3/4 |
- Material traps: construction finished in June

## rb12-summarization-hard-008

**Applies to:** routing-v1.2:summarization-hard-008

**Source:** The clinic lease was signed in February, renovation passed inspection in April, staff training finished in May, and opening is planned for July.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-008-r01` — Accurately state that the lease was signed in February. | 1/1 | The clinic lease was signed in February, renovation passed inspection in April, staff training finished in May, and opening is planned for July. | February | — |
| `rb12-summarization-hard-008-r02` — Accurately state that renovation passed inspection in April. | 1/1 | The clinic lease was signed in February, renovation passed inspection in April, staff training finished in May, and opening is planned for July. | April | — |
| `rb12-summarization-hard-008-r03` — Accurately state that staff training finished in May. | 1/1 | The clinic lease was signed in February, renovation passed inspection in April, staff training finished in May, and opening is planned for July. | May | — |
| `rb12-summarization-hard-008-r04` — Accurately state that opening is planned for July. | 1/1 | The clinic lease was signed in February, renovation passed inspection in April, staff training finished in May, and opening is planned for July. | July | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-008-r01` | 1 | 3/4 |
| `rb12-summarization-hard-008-r02` | 1 | 3/4 |
| `rb12-summarization-hard-008-r03` | 1 | 3/4 |
| `rb12-summarization-hard-008-r04` | 1 | 3/4 |
- Material traps: the clinic opened in May

## rb12-summarization-medium-011

**Applies to:** routing-v1.2:summarization-medium-011

**Source:** Option A cuts latency by 30% but raises cost by 15%. Option B keeps cost flat and cuts latency by 10%. The team selected B because the budget is fixed.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-011-r01a` — Option A cuts latency by 30%. | 1/2 | Option A cuts latency by 30% but raises cost by 15%. | 30%, 30, Option A | — |
| `rb12-summarization-medium-011-r01b` — Option A raises cost by 15%. | 1/2 | Option A cuts latency by 30% but raises cost by 15%. | 15%, 15, Option A | — |
| `rb12-summarization-medium-011-r02a` — Option B keeps cost flat. | 1/2 | Option B keeps cost flat and cuts latency by 10%. | Option B | — |
| `rb12-summarization-medium-011-r02b` — Option B cuts latency by 10%. | 1/2 | Option B keeps cost flat and cuts latency by 10%. | 10%, 10, Option B | — |
| `rb12-summarization-medium-011-r03` — Accurately state that the team selected B because the budget is fixed. | 1/1 | The team selected B because the budget is fixed. | — | — |

**Total weight:** 3; **minimum meaningful omission:** 1/2; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-011-r01a` | 1/2 | 5/6 |
| `rb12-summarization-medium-011-r01b` | 1/2 | 5/6 |
| `rb12-summarization-medium-011-r02a` | 1/2 | 5/6 |
| `rb12-summarization-medium-011-r02b` | 1/2 | 5/6 |
| `rb12-summarization-medium-011-r03` | 1 | 2/3 |
- Composite review: **SPLIT** — Accurately state that A cuts latency 30% but raises cost 15%. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Composite review: **SPLIT** — Accurately state that B keeps cost flat and cuts latency 10%. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Material traps: the team selected A

## rb12-summarization-hard-009

**Applies to:** routing-v1.2:summarization-hard-009

**Source:** Database X cuts storage cost 20% but increases recovery time from 10 to 35 minutes. Database Y keeps current cost and recovery time. The team retained Y because recovery speed is mandatory.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-009-r01` — Accurately state that X cuts storage cost 20%. | 1/1 | Database X cuts storage cost 20% but increases recovery time from 10 to 35 minutes. | 20%, 20 | — |
| `rb12-summarization-hard-009-r02a` — Database X has a current recovery time of 10 minutes. | 1/2 | Database X cuts storage cost 20% but increases recovery time from 10 to 35 minutes. | 10 minutes, Database X | — |
| `rb12-summarization-hard-009-r02b` — Database X would increase recovery time to 35 minutes. | 1/2 | Database X cuts storage cost 20% but increases recovery time from 10 to 35 minutes. | 35 minutes, Database X | CONDITIONAL: 35 minutes is a 25-minute increase from 10 minutes |
| `rb12-summarization-hard-009-r03` — Accurately state that Y preserves current cost and recovery time. | 1/1 | Database Y keeps current cost and recovery time. | — | — |
| `rb12-summarization-hard-009-r04` — Accurately state that the team retained Y for recovery speed. | 1/1 | The team retained Y because recovery speed is mandatory. | — | — |

**Total weight:** 4; **minimum meaningful omission:** 1/2; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-009-r01` | 1 | 3/4 |
| `rb12-summarization-hard-009-r02a` | 1/2 | 7/8 |
| `rb12-summarization-hard-009-r02b` | 1/2 | 7/8 |
| `rb12-summarization-hard-009-r03` | 1 | 3/4 |
| `rb12-summarization-hard-009-r04` | 1 | 3/4 |
- Ambiguity — old: Accurately state that X increases recovery time to 35 minutes.
  - Problem: The old wording named only the 35-minute endpoint although the source also made the 10-minute baseline material.
  - Resolution: Represent the baseline and new recovery time separately with half of the old weight each.
  - New propositions: rb12-summarization-hard-009-r02a, rb12-summarization-hard-009-r02b
  - Rationale: The resolution makes the benchmark-owned information boundary explicit.
- Material traps: the team selected X

## rb12-summarization-medium-012

**Applies to:** routing-v1.2:summarization-medium-012

**Source:** Support reproduced the defect and sent logs to Engineering. Engineering identified a parser bug and supplied a patch. Release Management scheduled the patch for Tuesday.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-012-r01a` — Support reproduced the defect. | 1/2 | Support reproduced the defect and sent logs to Engineering. | — | — |
| `rb12-summarization-medium-012-r01b` — Support sent logs to Engineering. | 1/2 | Support reproduced the defect and sent logs to Engineering. | — | — |
| `rb12-summarization-medium-012-r02a` — Engineering identified a parser bug. | 1/2 | Engineering identified a parser bug and supplied a patch. | — | — |
| `rb12-summarization-medium-012-r02b` — Engineering supplied a patch. | 1/2 | Engineering identified a parser bug and supplied a patch. | — | — |
| `rb12-summarization-medium-012-r03` — Accurately state that release management scheduled Tuesday. | 1/1 | Release Management scheduled the patch for Tuesday. | Tuesday | — |

**Total weight:** 3; **minimum meaningful omission:** 1/2; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-012-r01a` | 1/2 | 5/6 |
| `rb12-summarization-medium-012-r01b` | 1/2 | 5/6 |
| `rb12-summarization-medium-012-r02a` | 1/2 | 5/6 |
| `rb12-summarization-medium-012-r02b` | 1/2 | 5/6 |
| `rb12-summarization-medium-012-r03` | 1 | 2/3 |
- Composite review: **SPLIT** — Accurately state that support reproduced the defect and sent logs. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Composite review: **SPLIT** — Accurately state that engineering found a parser bug and supplied a patch. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Material traps: support supplied the patch

## rb12-summarization-medium-013

**Applies to:** routing-v1.2:summarization-medium-013

**Source:** Sales documented the contract exception and asked Legal for review. Legal approved revised language, then Operations added it to the renewal package due Friday.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-medium-013-r01a` — Sales documented the contract exception. | 1/2 | Sales documented the contract exception and asked Legal for review. | — | — |
| `rb12-summarization-medium-013-r01b` — Sales asked Legal for review. | 1/2 | Sales documented the contract exception and asked Legal for review. | — | — |
| `rb12-summarization-medium-013-r02` — Accurately state that Legal approved revised language. | 1/1 | Legal approved revised language, then Operations added it to the renewal package due Friday. | — | — |
| `rb12-summarization-medium-013-r03` — Accurately state that Operations added it to the Friday renewal package. | 1/1 | Legal approved revised language, then Operations added it to the renewal package due Friday. | Friday | — |

**Total weight:** 3; **minimum meaningful omission:** 1/2; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-medium-013-r01a` | 1/2 | 5/6 |
| `rb12-summarization-medium-013-r01b` | 1/2 | 5/6 |
| `rb12-summarization-medium-013-r02` | 1 | 2/3 |
| `rb12-summarization-medium-013-r03` | 1 | 2/3 |
- Composite review: **SPLIT** — Accurately state that Sales documented the exception and contacted Legal. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Material traps: Sales approved the legal language

## rb12-summarization-hard-010

**Applies to:** routing-v1.2:summarization-hard-010

**Source:** Revenue rose 6% and customer count rose 9%, while average order value fell 3%. Management attributed growth to new customers, not larger purchases.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-010-r01` — Accurately state that revenue rose 6%. | 1/1 | Revenue rose 6% and customer count rose 9%, while average order value fell 3%. | 6%, 6 | — |
| `rb12-summarization-hard-010-r02` — Accurately state that customer count rose 9%. | 1/1 | Revenue rose 6% and customer count rose 9%, while average order value fell 3%. | 9%, 9 | — |
| `rb12-summarization-hard-010-r03` — Accurately state that average order value fell 3%. | 1/1 | Revenue rose 6% and customer count rose 9%, while average order value fell 3%. | 3%, 3 | — |
| `rb12-summarization-hard-010-r04` — Accurately state that management attributed growth to new customers. | 1/1 | Management attributed growth to new customers, not larger purchases. | — | — |

**Total weight:** 4; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-010-r01` | 1 | 3/4 |
| `rb12-summarization-hard-010-r02` | 1 | 3/4 |
| `rb12-summarization-hard-010-r03` | 1 | 3/4 |
| `rb12-summarization-hard-010-r04` | 1 | 3/4 |
- Material traps: larger purchases drove growth

## rb12-summarization-hard-011

**Applies to:** routing-v1.2:summarization-hard-011

**Source:** Orders grew 11% and delivery time improved 8%, but returns rose from 4% to 6%. Leaders credited warehouse automation for speed while opening a review of return causes.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `rb12-summarization-hard-011-r01` — Accurately state that orders grew 11%. | 1/1 | Orders grew 11% and delivery time improved 8%, but returns rose from 4% to 6%. | 11%, 11 | — |
| `rb12-summarization-hard-011-r02` — Accurately state that delivery time improved 8%. | 1/1 | Orders grew 11% and delivery time improved 8%, but returns rose from 4% to 6%. | 8%, 8 | — |
| `rb12-summarization-hard-011-r03` — Accurately state that returns rose from 4% to 6%. | 1/1 | Orders grew 11% and delivery time improved 8%, but returns rose from 4% to 6%. | 4%, 6%, 4, 6 | NOT_ALLOWED: returns rising from 4% to 6% = a 2-point increase |
| `rb12-summarization-hard-011-r04a` — Leaders credited warehouse automation for improved delivery speed. | 1/2 | Leaders credited warehouse automation for speed while opening a review of return causes. | — | — |
| `rb12-summarization-hard-011-r04b` — Leaders opened a review of the causes of increased returns. | 1/2 | Leaders credited warehouse automation for speed while opening a review of return causes. | — | — |

**Total weight:** 4; **minimum meaningful omission:** 1/2; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `rb12-summarization-hard-011-r01` | 1 | 3/4 |
| `rb12-summarization-hard-011-r02` | 1 | 3/4 |
| `rb12-summarization-hard-011-r03` | 1 | 3/4 |
| `rb12-summarization-hard-011-r04a` | 1/2 | 7/8 |
| `rb12-summarization-hard-011-r04b` | 1/2 | 7/8 |
- Composite review: **SPLIT** — Accurately state that leaders credited automation for speed and are reviewing returns. — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Ambiguity — old: Accurately state that leaders credited automation for speed and are reviewing returns.
  - Problem: The old requirement combined an attribution about speed with a separate review of return causes.
  - Resolution: Represent the attribution and review as independent half-weight propositions.
  - New propositions: rb12-summarization-hard-011-r04a, rb12-summarization-hard-011-r04b
  - Rationale: The resolution makes the benchmark-owned information boundary explicit.
- Material traps: automation was proven to cause the higher return rate

## foundation-shared-summarization-easy-01

**Applies to:** foundation-v2:summarization-easy-01, foundation-v3:summarization-easy-01

**Source:** The North Clinic will open two hours late on Friday because of electrical maintenance. Emergency services remain available.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `foundation-shared-summarization-easy-01-r01` — Friday opening is delayed by two hours. | 1/1 | The North Clinic will open two hours late on Friday because of electrical maintenance. | Friday | — |
| `foundation-shared-summarization-easy-01-r02` — electrical maintenance is the cause. | 1/1 | The North Clinic will open two hours late on Friday because of electrical maintenance. | — | — |
| `foundation-shared-summarization-easy-01-r03` — emergency services remain available. | 1/1 | Emergency services remain available. | — | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `foundation-shared-summarization-easy-01-r01` | 1 | 2/3 |
| `foundation-shared-summarization-easy-01-r02` | 1 | 2/3 |
| `foundation-shared-summarization-easy-01-r03` | 1 | 2/3 |

## foundation-shared-summarization-easy-02

**Applies to:** foundation-v2:summarization-easy-02, foundation-v3:summarization-easy-02

**Source:** The garden workshop moved from Room 2 to Room 5. Its 3 p.m. start time is unchanged.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `foundation-shared-summarization-easy-02-r01` — workshop moved from Room 2 to Room 5. | 1/1 | The garden workshop moved from Room 2 to Room 5. | 2, 5, Room 2, Room 5 | — |
| `foundation-shared-summarization-easy-02-r02` — start remains 3 p.m. | 1/1 | Its 3 p.m. | 3 p.m., 3 | — |

**Total weight:** 2; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `foundation-shared-summarization-easy-02-r01` | 1 | 1/2 |
| `foundation-shared-summarization-easy-02-r02` | 1 | 1/2 |

## foundation-shared-summarization-medium-01

**Applies to:** foundation-v2:summarization-medium-01, foundation-v3:summarization-medium-01

**Source:** River buses pause Monday morning for inspection and resume at noon. Rail service is unaffected. The cafe opens normally; yesterday's rain total was 8 mm.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `foundation-shared-summarization-medium-01-r01` — river buses pause Monday morning. | 1/1 | River buses pause Monday morning for inspection and resume at noon. | Monday | — |
| `foundation-shared-summarization-medium-01-r02` — service resumes at noon. | 1/1 | River buses pause Monday morning for inspection and resume at noon. | noon | — |
| `foundation-shared-summarization-medium-01-r03` — rail service is unaffected. | 1/1 | Rail service is unaffected. | polarity/constraint | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `foundation-shared-summarization-medium-01-r01` | 1 | 2/3 |
| `foundation-shared-summarization-medium-01-r02` | 1 | 2/3 |
| `foundation-shared-summarization-medium-01-r03` | 1 | 2/3 |
- Material traps: cafe closure

## foundation-shared-summarization-medium-02

**Applies to:** foundation-v2:summarization-medium-02, foundation-v3:summarization-medium-02

**Source:** The trial enrolled 40 volunteers. Thirty-six completed it. Reported wait time fell from 12 to 8 minutes. The office wallpaper was replaced in June.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `foundation-shared-summarization-medium-02-r01a` — The trial enrolled 40 volunteers. | 1/2 | The trial enrolled 40 volunteers. | 40 | — |
| `foundation-shared-summarization-medium-02-r01b` — Thirty-six volunteers completed the trial. | 1/2 | Thirty-six completed it. | — | — |
| `foundation-shared-summarization-medium-02-r02` — wait time fell from 12 to 8 minutes. | 1/1 | Reported wait time fell from 12 to 8 minutes. | 12, 8 minutes | NOT_ALLOWED: 12 to 8 minutes = a 4-minute decrease |

**Total weight:** 2; **minimum meaningful omission:** 1/2; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `foundation-shared-summarization-medium-02-r01a` | 1/2 | 3/4 |
| `foundation-shared-summarization-medium-02-r01b` | 1/2 | 3/4 |
| `foundation-shared-summarization-medium-02-r02` | 1 | 1/2 |
- Composite review: **SPLIT** — 40 enrolled and 36 completed — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.

## foundation-shared-summarization-medium-03

**Applies to:** foundation-v2:summarization-medium-03, foundation-v3:summarization-medium-03

**Source:** Cedar School adds a morning bus on Route 3 from September 4; the afternoon schedule and fares remain unchanged.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `foundation-shared-summarization-medium-03-r01` — morning bus added to Route 3. | 1/1 | Cedar School adds a morning bus on Route 3 from September 4; the afternoon schedule and fares remain unchanged. | 3, Route 3 | — |
| `foundation-shared-summarization-medium-03-r02` — starts September 4. | 1/1 | Cedar School adds a morning bus on Route 3 from September 4; the afternoon schedule and fares remain unchanged. | September 4, 4 | — |
| `foundation-shared-summarization-medium-03-r03` — afternoon schedule and fares unchanged. | 1/1 | Cedar School adds a morning bus on Route 3 from September 4; the afternoon schedule and fares remain unchanged. | polarity/constraint | — |

**Total weight:** 3; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `foundation-shared-summarization-medium-03-r01` | 1 | 2/3 |
| `foundation-shared-summarization-medium-03-r02` | 1 | 2/3 |
| `foundation-shared-summarization-medium-03-r03` | 1 | 2/3 |

## foundation-shared-summarization-hard-01

**Applies to:** foundation-v2:summarization-hard-01, foundation-v3:summarization-hard-01

**Source:** A six-month pilot gave 120 homes smart meters; 108 supplied complete data. Average electricity use fell 9%, but the report cannot attribute causation. Installation cost exceeded plan by 4%. A separate survey covered paint colors.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `foundation-shared-summarization-hard-01-r01a` — The pilot included 120 homes. | 1/2 | A six-month pilot gave 120 homes smart meters; 108 supplied complete data. | 120 homes | — |
| `foundation-shared-summarization-hard-01-r01b` — One hundred eight homes supplied complete data. | 1/2 | A six-month pilot gave 120 homes smart meters; 108 supplied complete data. | — | CONDITIONAL: 108 of 120 homes = 90% complete data |
| `foundation-shared-summarization-hard-01-r02` — average use fell 9%. | 1/1 | Average electricity use fell 9%, but the report cannot attribute causation. | 9%, 9 | — |
| `foundation-shared-summarization-hard-01-r03` — The report does not establish that smart meters caused the 9% decline in electricity use. | 1/1 | Average electricity use fell 9%, but the report cannot attribute causation. | 9%, 9, polarity/constraint | — |
| `foundation-shared-summarization-hard-01-r04` — installation cost exceeded plan by 4%. | 1/1 | Installation cost exceeded plan by 4%. | 4%, 4 | — |

**Total weight:** 4; **minimum meaningful omission:** 1/2; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `foundation-shared-summarization-hard-01-r01a` | 1/2 | 7/8 |
| `foundation-shared-summarization-hard-01-r01b` | 1/2 | 7/8 |
| `foundation-shared-summarization-hard-01-r02` | 1 | 3/4 |
| `foundation-shared-summarization-hard-01-r03` | 1 | 3/4 |
| `foundation-shared-summarization-hard-01-r04` | 1 | 3/4 |
- Composite review: **SPLIT** — 120 homes participated and 108 supplied complete data — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Ambiguity — old: causation is not established
  - Problem: The old wording did not identify which causal relationship was unestablished.
  - Resolution: State explicitly that the report does not establish that smart meters caused the 9% decline.
  - New propositions: foundation-shared-summarization-hard-01-r03
  - Rationale: The resolution makes the benchmark-owned information boundary explicit.
- Material traps: proved the meters caused

## foundation-shared-summarization-hard-02

**Applies to:** foundation-v2:summarization-hard-02, foundation-v3:summarization-hard-02

**Source:** After a sensor fault, Harbor Plant recalled lots H21 and H22. Lot H20 passed retesting. No injuries were reported. Replacement shipments begin May 6, prioritizing hospitals. Quarterly sales rose 2%, an unrelated fact.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `foundation-shared-summarization-hard-02-r01` — H21 and H22 recalled after sensor fault. | 1/1 | After a sensor fault, Harbor Plant recalled lots H21 and H22. | H21, H22 | — |
| `foundation-shared-summarization-hard-02-r02` — H20 passed retesting. | 1/1 | Lot H20 passed retesting. | H20 | — |
| `foundation-shared-summarization-hard-02-r03` — no injuries. | 1/1 | No injuries were reported. | polarity/constraint | — |
| `foundation-shared-summarization-hard-02-r04a` — Replacement shipments begin May 6. | 1/2 | Replacement shipments begin May 6, prioritizing hospitals. | May 6, 6 | — |
| `foundation-shared-summarization-hard-02-r04b` — Hospitals receive replacement priority. | 1/2 | Replacement shipments begin May 6, prioritizing hospitals. | — | — |

**Total weight:** 4; **minimum meaningful omission:** 1/2; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `foundation-shared-summarization-hard-02-r01` | 1 | 3/4 |
| `foundation-shared-summarization-hard-02-r02` | 1 | 3/4 |
| `foundation-shared-summarization-hard-02-r03` | 1 | 3/4 |
| `foundation-shared-summarization-hard-02-r04a` | 1/2 | 7/8 |
| `foundation-shared-summarization-hard-02-r04b` | 1/2 | 7/8 |
- Composite review: **RETAIN** — H21 and H22 recalled after sensor fault — The affected lots and sensor-fault context form one recall event.
- Composite review: **SPLIT** — replacement begins May 6 with hospitals prioritized — The components are independently meaningful; equal fractional weights preserve the old requirement's total weight.
- Material traps: H20 recalled

## foundation-shared-summarization-hard-03

**Applies to:** foundation-v2:summarization-hard-03, foundation-v3:summarization-hard-03

**Source:** The council approved night repairs on Bridge K from July 8-11, closing one lane from 10 p.m. to 5 a.m. Buses remain on schedule. Emergency vehicles get priority. A proposed daytime closure was rejected.

| Proposition | Weight | Evidence | Anchors | Derivation |
|---|---:|---|---|---|
| `foundation-shared-summarization-hard-03-r01` — night repairs July 8-11. | 1/1 | The council approved night repairs on Bridge K from July 8-11, closing one lane from 10 p.m. | July 8-11, 8, 11 | — |
| `foundation-shared-summarization-hard-03-r02` — one lane closed 10 p.m.-5 a.m. | 1/1 | The council approved night repairs on Bridge K from July 8-11, closing one lane from 10 p.m. | 10 p.m., 5 a.m., 10, 5 | — |
| `foundation-shared-summarization-hard-03-r03` — buses remain on schedule. | 1/1 | Buses remain on schedule. | — | — |
| `foundation-shared-summarization-hard-03-r04` — emergency vehicles have priority. | 1/1 | Emergency vehicles get priority. | — | — |
| `foundation-shared-summarization-hard-03-r05` — daytime closure rejected. | 1/1 | A proposed daytime closure was rejected. | — | — |

**Total weight:** 5; **minimum meaningful omission:** 1; the table below audits every single-proposition omission.

| Omitted proposition | Omitted weight | Resulting coverage |
|---|---:|---:|
| `foundation-shared-summarization-hard-03-r01` | 1 | 4/5 |
| `foundation-shared-summarization-hard-03-r02` | 1 | 4/5 |
| `foundation-shared-summarization-hard-03-r03` | 1 | 4/5 |
| `foundation-shared-summarization-hard-03-r04` | 1 | 4/5 |
| `foundation-shared-summarization-hard-03-r05` | 1 | 4/5 |
- Material traps: daytime closure approved

## Frozen 24-case validation set

Expected labels are stored separately from judge inputs.

| Case | Phenomenon | Expected | Source | Proposition | Candidate |
|---|---|---|---|---|---|
| prop-live-01 | DIRECT_ENTAILMENT | ENTAILED | The archive opens at 08:30 on Tuesday. | The archive opens Tuesday at 08:30. | The archive will open Tuesday at 8:30 a.m. |
| prop-live-02 | DIRECT_ENTAILMENT | ENTAILED | Mara approved the revised budget. | Mara approved the revised budget. | Mara approved the revised budget. |
| prop-live-03 | LEXICAL_PARAPHRASE | ENTAILED | The board postponed the vote until June. | The vote was postponed until June. | The board delayed its vote to June. |
| prop-live-04 | LEXICAL_PARAPHRASE | ENTAILED | Rail service was unaffected by the inspection. | Rail service was unaffected. | The inspection did not disrupt trains. |
| prop-live-05 | SEMANTIC_COMPRESSION | ENTAILED | Northbound trains stopped at 08:10 and resumed at 08:25. | Northbound service was interrupted. | Northbound service briefly paused. |
| prop-live-06 | SEMANTIC_COMPRESSION | ENTAILED | The clinic received no deliveries for three days and canceled appointments. | The delivery interruption affected appointments. | A three-day delivery gap disrupted appointments. |
| prop-live-07 | SEMANTIC_COMPRESSION | ENTAILED | The same invoice was entered twice; staff removed the second entry. | Staff removed the duplicate invoice entry. | Staff deleted the duplicated invoice record. |
| prop-live-08 | VALID_DERIVATION | ENTAILED | Of 200 registered participants, 150 attended. | Three quarters of registered participants attended. | Attendance was 75% of registrations. |
| prop-live-09 | VALID_DERIVATION | ENTAILED | The price fell from $20 to $15. | The price decreased by $5. | The item became five dollars cheaper. |
| prop-live-10 | VALID_DERIVATION | ENTAILED | The outage began at 11:05 and ended at 11:35. | The outage lasted 30 minutes. | Service was down for half an hour. |
| prop-live-11 | INVALID_DERIVATION | NOT_ENTAILED | The price fell from $20 to $15. | The final price was $15. | The price decreased by $5. |
| prop-live-12 | INVALID_DERIVATION | NOT_ENTAILED | Three of six samples passed. | Exactly three samples passed. | Half of the samples passed. |
| prop-live-13 | INVALID_DERIVATION | NOT_ENTAILED | Restrictions applied Monday through Wednesday. | Restrictions ended Wednesday. | Restrictions lasted three days. |
| prop-live-14 | MISSING_FACT | NOT_ENTAILED | The permit was approved Thursday after a diagram arrived Wednesday. | The diagram arrived Wednesday. | The permit was approved Thursday. |
| prop-live-15 | MISSING_FACT | NOT_ENTAILED | The shipment contained rice and medical supplies. | The shipment contained medical supplies. | The shipment contained rice. |
| prop-live-16 | CONTRADICTION | NOT_ENTAILED | The east entrance remained closed throughout the event. | The east entrance remained closed. | The east entrance stayed open during the event. |
| prop-live-17 | CONTRADICTION | NOT_ENTAILED | No customer records were lost. | No customer records were lost. | Some customer records were lost. |
| prop-live-18 | UNSUPPORTED_CAUSATION | NOT_ENTAILED | A patch was installed Monday. Errors declined Tuesday. | The patch caused errors to decline. | Monday's patch caused Tuesday's decline in errors. |
| prop-live-19 | UNSUPPORTED_CAUSATION | NOT_ENTAILED | The alarm sounded before staff evacuated. | The alarm caused the evacuation. | Staff evacuated because the alarm sounded. |
| prop-live-20 | UNSUPPORTED_CAUSATION | AMBIGUOUS | A fan stopped and the server overheated moments later. | The stopped fan caused the overheating. | The fan failure led to overheating. |
| prop-live-21 | ATTRIBUTION_CHANGE | NOT_ENTAILED | Mina proposed the amendment; the committee approved it. | Mina proposed the amendment. | The committee proposed the amendment. |
| prop-live-22 | ATTRIBUTION_CHANGE | AMBIGUOUS | The laboratory issued the report after Ana completed the analysis. | Ana authored the report. | Ana's report was issued by the laboratory. |
| prop-live-23 | CERTAINTY_CHANGE | NOT_ENTAILED | Analysts said demand may rise. | Analysts expressed uncertainty about rising demand. | Analysts said demand will rise. |
| prop-live-24 | CERTAINTY_CHANGE | AMBIGUOUS | The pilot appears successful on the available measures. | The pilot was successful. | The successful pilot met the available measures. |

Historical Astra 1.1 and 1.2 expected labels remain unchanged. The earlier descriptive-characterization and duration-compression expectations are annotated as underspecified historical fixtures; they are not relabeled here.
