"""Benchmark-owned summarization proposition specification; no evaluator calls."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from adaptive_llm_gateway.models.schemas import DomainModel

from .models import BenchmarkDataset, BenchmarkTask, load_dataset

SPECIFICATION_VERSION = "1.0.0"
ROUTING_DATASET_VERSION = "1.2.0"
VALIDATION_SET_VERSION = "1.0.0"

ROUTING_V11 = Path("benchmarks/datasets/routing-benchmark-v1.json")
ROUTING_V12 = Path("benchmarks/datasets/routing-benchmark-v1.2.json")
FOUNDATION_V2 = Path("benchmarks/datasets/foundation-v2.json")
FOUNDATION_V3 = Path("benchmarks/datasets/foundation-v3.json")
SPEC_PATH = Path("benchmarks/specifications/summarization-propositions-v1.0.0.json")
VALIDATION_INPUT_PATH = Path(
    "benchmarks/validation/summarization-proposition-validation-v1.0.0.json")
VALIDATION_LABEL_PATH = Path(
    "benchmarks/validation/summarization-proposition-validation-v1.0.0-labels.json")
REVIEW_PATH = Path("docs/phase-9-summarization-proposition-review.md")


class RequirementType(StrEnum):
    ATOMIC = "atomic"
    COHESIVE_COMPOSITE = "cohesive_composite"


class EntailmentVerdict(StrEnum):
    ENTAILED = "ENTAILED"
    NOT_ENTAILED = "NOT_ENTAILED"
    AMBIGUOUS = "AMBIGUOUS"


class DerivationPolicy(StrEnum):
    ALLOWED = "ALLOWED"
    NOT_ALLOWED = "NOT_ALLOWED"
    CONDITIONAL = "CONDITIONAL"


class AnchorKind(StrEnum):
    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    TIME = "time"
    NAMED_ENTITY = "named_entity"
    BOOLEAN_STATE = "boolean_state"
    EXPLICIT_LABEL = "explicit_label"


class MaterialErrorCategory(StrEnum):
    UNSUPPORTED_MATERIAL_CLAIM = "UNSUPPORTED_MATERIAL_CLAIM"
    CONTRADICTION = "CONTRADICTION"
    UNSUPPORTED_CAUSATION = "UNSUPPORTED_CAUSATION"
    ATTRIBUTION_ERROR = "ATTRIBUTION_ERROR"
    QUANTITY_OR_TIME_ERROR = "QUANTITY_OR_TIME_ERROR"
    CERTAINTY_DISTORTION = "CERTAINTY_DISTORTION"


class RationalWeight(DomainModel):
    numerator: int = Field(gt=0, strict=True)
    denominator: int = Field(gt=0, strict=True)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


class SourceEvidence(DomainModel):
    quote: str = Field(min_length=1)


class DeterministicAnchor(DomainModel):
    kind: AnchorKind
    value: str = Field(min_length=1)
    normalized_value: str = Field(min_length=1)
    required_exactly: bool = False


class DirectionalEquivalence(DomainModel):
    alternative: str = Field(min_length=1)
    direction: Literal["alternative_satisfies_proposition"] = "alternative_satisfies_proposition"
    preserves_information: str = Field(min_length=1)


class DerivationRule(DomainModel):
    derivation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    policy: DerivationPolicy
    expression: str = Field(min_length=1)
    requires_proposition_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def conditional_rules_declare_dependencies(self):
        if self.policy is DerivationPolicy.CONDITIONAL and not self.requires_proposition_ids:
            raise ValueError("conditional derivations require proposition dependencies")
        if self.policy is not DerivationPolicy.CONDITIONAL and self.requires_proposition_ids:
            raise ValueError("only conditional derivations may declare dependencies")
        return self


class SemanticProposition(DomainModel):
    proposition_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    legacy_requirement_index: int = Field(ge=1, strict=True)
    description: str = Field(min_length=1)
    requirement_type: RequirementType
    weight: RationalWeight
    source_evidence: SourceEvidence
    deterministic_anchors: tuple[DeterministicAnchor, ...] = ()
    accepted_equivalences: tuple[DirectionalEquivalence, ...] = ()
    derivations: tuple[DerivationRule, ...] = ()
    notes: str = ""


class CompositeReview(DomainModel):
    legacy_requirement_index: int = Field(ge=1, strict=True)
    old_requirement: str = Field(min_length=1)
    decision: Literal["split", "retain"]
    resulting_proposition_ids: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class AmbiguityResolution(DomainModel):
    old_requirement: str = Field(min_length=1)
    ambiguity: str = Field(min_length=1)
    resolution: str = Field(min_length=1)
    new_proposition_ids: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class SummarizationTaskSpecification(DomainModel):
    specification_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    canonical_task_id: str
    applies_to: tuple[str, ...] = Field(min_length=1)
    source_text: str = Field(min_length=1)
    legacy_requirements: tuple[str, ...] = Field(min_length=1)
    propositions: tuple[SemanticProposition, ...] = Field(min_length=1)
    composite_reviews: tuple[CompositeReview, ...] = ()
    ambiguity_resolutions: tuple[AmbiguityResolution, ...] = ()
    material_semantic_traps: tuple[str, ...] = ()

    @model_validator(mode="after")
    def stable_unique_weighted_propositions(self):
        ids = [item.proposition_id for item in self.propositions]
        if len(ids) != len(set(ids)):
            raise ValueError("proposition IDs must be unique within a task")
        if sum((item.weight.fraction for item in self.propositions), Fraction()) != len(
                self.legacy_requirements):
            raise ValueError("splits must preserve the legacy task's total semantic weight")
        return self


class MaterialErrorFinding(DomainModel):
    candidate_claim: str = Field(min_length=1)
    source_evidence: SourceEvidence
    category: MaterialErrorCategory
    materiality: Literal["material", "borderline"]
    reason: str = Field(min_length=1)


class MaterialErrorContract(DomainModel):
    required_fields: tuple[str, ...] = (
        "candidate_claim", "source_evidence", "category", "materiality", "reason")
    categories: tuple[MaterialErrorCategory, ...] = tuple(MaterialErrorCategory)
    interpretive_policy: str = (
        "Clearly supported characterizations are not errors; clearly unsupported substantive "
        "assertions are material errors; genuinely borderline interpretations are marked "
        "borderline and produce an AMBIGUOUS evaluation requiring review.")


class PropositionSpecification(DomainModel):
    specification_version: str = SPECIFICATION_VERSION
    source_benchmarks: tuple[str, ...]
    tasks: tuple[SummarizationTaskSpecification, ...]
    material_error_categories: tuple[MaterialErrorCategory, ...] = tuple(MaterialErrorCategory)
    material_error_contract: MaterialErrorContract = Field(default_factory=MaterialErrorContract)

    @model_validator(mode="after")
    def unique_specification_and_proposition_ids(self):
        task_ids = [item.specification_id for item in self.tasks]
        proposition_ids = [p.proposition_id for task in self.tasks for p in task.propositions]
        if len(task_ids) != len(set(task_ids)) or len(proposition_ids) != len(set(proposition_ids)):
            raise ValueError("specification and proposition IDs must be globally unique")
        return self


class CoverageResult(DomainModel):
    status: Literal["complete", "incomplete"]
    coverage_numerator: int | None = None
    coverage_denominator: int | None = None
    ambiguous_proposition_ids: tuple[str, ...] = ()


def weighted_coverage(task: SummarizationTaskSpecification,
                      verdicts: dict[str, EntailmentVerdict]) -> CoverageResult:
    expected = {item.proposition_id for item in task.propositions}
    if set(verdicts) != expected:
        raise ValueError("verdicts must cover every proposition exactly once")
    ambiguous = tuple(sorted(key for key, value in verdicts.items()
                             if value is EntailmentVerdict.AMBIGUOUS))
    if ambiguous:
        return CoverageResult(status="incomplete", ambiguous_proposition_ids=ambiguous)
    weights = {item.proposition_id: item.weight.fraction for item in task.propositions}
    earned = sum((weights[key] for key, value in verdicts.items()
                  if value is EntailmentVerdict.ENTAILED), Fraction())
    total = sum(weights.values(), Fraction())
    coverage = earned / total
    return CoverageResult(status="complete", coverage_numerator=coverage.numerator,
                          coverage_denominator=coverage.denominator)


class PropositionJudgeInput(DomainModel):
    case_id: str = Field(pattern=r"^prop-live-[0-9]{2}$")
    source_evidence: str = Field(min_length=1)
    required_proposition: str = Field(min_length=1)
    candidate_text: str = Field(min_length=1)


class ValidationExpectation(DomainModel):
    expected_verdict: EntailmentVerdict
    phenomenon: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


# Manually reviewed composites. Foundation definitions are canonical and reused by V2/V3.
_SPLITS: dict[tuple[str, str, int], tuple[str, ...]] = {
    ("routing", "summarization-easy-007", 1): (
        "A cooling fan failed.", "The fan failure caused the server to shut down."),
    ("routing", "summarization-medium-008", 1): (
        "Plan North costs $18 monthly.", "Plan North includes 20 GB."),
    ("routing", "summarization-medium-008", 2): (
        "Plan South costs $24 monthly.", "Plan South includes 50 GB.",
        "Plan South includes roaming."),
    ("routing", "summarization-medium-009", 1): (
        "Vendor Pine charges a $900 setup fee.", "Vendor Pine charges $80 monthly."),
    ("routing", "summarization-medium-009", 3): (
        "Vendor Lake has no setup fee.", "Vendor Lake charges $125 monthly."),
    ("routing", "summarization-medium-011", 1): (
        "Option A cuts latency by 30%.", "Option A raises cost by 15%."),
    ("routing", "summarization-medium-011", 2): (
        "Option B keeps cost flat.", "Option B cuts latency by 10%."),
    ("routing", "summarization-medium-012", 1): (
        "Support reproduced the defect.", "Support sent logs to Engineering."),
    ("routing", "summarization-medium-012", 2): (
        "Engineering identified a parser bug.", "Engineering supplied a patch."),
    ("routing", "summarization-medium-013", 1): (
        "Sales documented the contract exception.", "Sales asked Legal for review."),
    ("routing", "summarization-hard-011", 4): (
        "Leaders credited warehouse automation for improved delivery speed.",
        "Leaders opened a review of the causes of increased returns."),
    ("routing", "summarization-hard-009", 2): (
        "Database X has a current recovery time of 10 minutes.",
        "Database X would increase recovery time to 35 minutes."),
    ("foundation", "summarization-medium-02", 1): (
        "The trial enrolled 40 volunteers.", "Thirty-six volunteers completed the trial."),
    ("foundation", "summarization-hard-01", 1): (
        "The pilot included 120 homes.", "One hundred eight homes supplied complete data."),
    ("foundation", "summarization-hard-02", 4): (
        "Replacement shipments begin May 6.", "Hospitals receive replacement priority."),
}

_RETAINED_COMPOSITES = {("foundation", "summarization-hard-02", 1)}
_AMBIGUITY_ONLY_SPLITS = {("routing", "summarization-hard-009", 2)}

_AMBIGUITIES = {
    ("routing", "summarization-hard-009", 2): (
        "The old wording named only the 35-minute endpoint although the source also made the 10-minute baseline material.",
        "Represent the baseline and new recovery time separately with half of the old weight each."),
    ("routing", "summarization-hard-011", 4): (
        "The old requirement combined an attribution about speed with a separate review of return causes.",
        "Represent the attribution and review as independent half-weight propositions."),
    ("foundation", "summarization-hard-01", 3): (
        "The old wording did not identify which causal relationship was unestablished.",
        "State explicitly that the report does not establish that smart meters caused the 9% decline."),
}

_DESCRIPTION_OVERRIDES = {
    ("foundation", "summarization-hard-01", 3):
        "The report does not establish that smart meters caused the 9% decline in electricity use.",
}

_DERIVATION_TEMPLATES: dict[tuple[str, str, int, str], tuple[DerivationPolicy, str, tuple[tuple[int, str], ...], str]] = {
    ("routing", "summarization-medium-002", 1, ""):
        (DerivationPolicy.ALLOWED, "12,400 people = 12.4 thousand people", (),
         "Decimal scale notation preserves the same absolute reach."),
    ("routing", "summarization-medium-002", 4, ""):
        (DerivationPolicy.ALLOWED, "$1,550 = $1.55 thousand", (),
         "Currency scale notation preserves the same spend."),
    ("routing", "summarization-medium-003", 2, ""):
        (DerivationPolicy.CONDITIONAL, "360 of 480 = 75% attendance", ((1, ""),),
         "The percentage preserves attendance only when the registration base is also entailed."),
    ("routing", "summarization-medium-005", 2, ""):
        (DerivationPolicy.CONDITIONAL, "20 days + 5 days = a 25-day leave deadline", ((1, ""),),
         "The derived deadline is valid only when the standard 20-day basis is preserved."),
    ("routing", "summarization-medium-006", 2, ""):
        (DerivationPolicy.CONDITIONAL, "110 TB is a 30 TB increase from 80 TB", ((1, ""),),
         "A relative increase requires the starting target."),
    ("routing", "summarization-medium-006", 3, ""):
        (DerivationPolicy.CONDITIONAL, "95 TB is a 15 TB reduction from 110 TB", ((2, ""),),
         "A relative reduction requires the preceding target."),
    ("routing", "summarization-hard-009", 2, "b"):
        (DerivationPolicy.CONDITIONAL, "35 minutes is a 25-minute increase from 10 minutes", ((2, "a"),),
         "The delta preserves the endpoint only when the baseline is also entailed."),
    ("foundation", "summarization-medium-02", 2, ""):
        (DerivationPolicy.NOT_ALLOWED, "12 to 8 minutes = a 4-minute decrease", (),
         "A delta alone loses both endpoint measurements, which are benchmark material."),
    ("foundation", "summarization-hard-01", 1, "b"):
        (DerivationPolicy.CONDITIONAL, "108 of 120 homes = 90% complete data", ((1, "a"),),
         "The ratio preserves the completion count only when the 120-home base is entailed."),
    ("routing", "summarization-hard-011", 3, ""):
        (DerivationPolicy.NOT_ALLOWED, "returns rising from 4% to 6% = a 2-point increase", (),
         "The delta alone loses both return-rate endpoints."),
}


def _slug(prefix: str, task_id: str, index: int, suffix: str = "") -> str:
    return f"{prefix}-{task_id}-r{index:02d}{suffix}"


def _source_text(task: BenchmarkTask) -> str:
    marker = "Source:"
    if marker in task.prompt:
        return task.prompt.split(marker, 1)[1].strip()
    return task.prompt.split(":", 1)[1].strip() if ":" in task.prompt else task.prompt


_STOP = {"accurately", "state", "that", "the", "a", "an", "is", "was", "were",
         "to", "and", "for", "of", "it", "its", "with", "from", "by", "in"}


def _tokens(value: str) -> set[str]:
    return {item for item in re.findall(r"[a-z0-9]+", value.casefold()) if item not in _STOP}


def _evidence(source: str, description: str) -> str:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", source) if item.strip()]
    target = _tokens(description)
    return max(sentences, key=lambda item: (len(target & _tokens(item)), -len(item)))


def _anchors(description: str) -> tuple[DeterministicAnchor, ...]:
    found: list[DeterministicAnchor] = []
    patterns = (
        (AnchorKind.CURRENCY, r"\$[0-9][0-9,]*(?:\.[0-9]+)?"),
        (AnchorKind.PERCENTAGE, r"\b[0-9]+(?:\.[0-9]+)?%"),
        (AnchorKind.TIME, r"\b(?:[0-2]?[0-9]:[0-5][0-9]|noon|[0-9]+\s*(?:a\.m\.|p\.m\.))"),
        (AnchorKind.DATE, r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+[0-9]+(?:-[0-9]+)?)?"),
        (AnchorKind.NUMBER, r"\b[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*(?:TB|GB|days?|minutes?|homes?|people))?"),
    )
    occupied: set[str] = set()
    for kind, pattern in patterns:
        for match in re.finditer(pattern, description, flags=re.IGNORECASE):
            value = match.group(0)
            normalized = value.casefold().replace(",", "").replace(" ", "")
            key = normalized
            if key not in occupied:
                occupied.add(key)
                found.append(DeterministicAnchor(kind=kind, value=value,
                    normalized_value=normalized, required_exactly=False))
    if re.search(r"\b(?:no|not|unchanged|unaffected|never|only|must|requires?)\b",
                 description, flags=re.IGNORECASE):
        found.append(DeterministicAnchor(kind=AnchorKind.BOOLEAN_STATE,
            value="polarity/constraint", normalized_value="semantic_boolean", required_exactly=False))
    explicit_labels = re.findall(
        r"\b(?:H[0-9]+|Route\s+[0-9]+|Room\s+[0-9]+|Bridge\s+[A-Z]|Option\s+[A-Z]|"
        r"Database\s+[A-Z]|Plan\s+(?:North|South)|Vendor\s+(?:Pine|Lake))\b",
        description)
    for value in explicit_labels:
        normalized = re.sub(r"\s+", "", value.casefold())
        if normalized not in occupied:
            occupied.add(normalized)
            found.append(DeterministicAnchor(kind=AnchorKind.EXPLICIT_LABEL, value=value,
                normalized_value=normalized, required_exactly=False))
    named_entities = re.findall(
        r"\b(?:Asha|Bo|Cy|Nora|Omar|Pia|Maya|Rafi|Chen|Inez|Jordan|Priya|Luis|Morgan|"
        r"Finance|Harbor Plant|Cedar School|North Clinic|Release Management)\b",
        description)
    for value in named_entities:
        normalized = re.sub(r"\s+", "", value.casefold())
        if normalized not in occupied:
            occupied.add(normalized)
            found.append(DeterministicAnchor(kind=AnchorKind.NAMED_ENTITY, value=value,
                normalized_value=normalized, required_exactly=False))
    return tuple(found)


def _task_spec(task: BenchmarkTask, namespace: str) -> SummarizationTaskSpecification:
    family = "routing" if namespace == "rb12" else "foundation"
    prefix = "rb12" if family == "routing" else "foundation-shared"
    source = _source_text(task)
    requirements = tuple(str(item) for item in task.evaluation_metadata["semantic_requirements"])
    propositions: list[SemanticProposition] = []
    reviews: list[CompositeReview] = []
    ambiguity: list[AmbiguityResolution] = []
    ids_by_legacy: dict[tuple[int, str], str] = {}
    pending: list[tuple[int, str, str, RationalWeight, RequirementType]] = []
    for index, old in enumerate(requirements, 1):
        key = (family, task.task_id, index)
        children = _SPLITS.get(key)
        if children:
            count = len(children)
            child_ids = []
            for offset, description in enumerate(children):
                suffix = chr(ord("a") + offset)
                prop_id = _slug(prefix, task.task_id, index, suffix)
                ids_by_legacy[(index, suffix)] = prop_id
                child_ids.append(prop_id)
                pending.append((index, suffix, description,
                    RationalWeight(numerator=1, denominator=count), RequirementType.ATOMIC))
            if key not in _AMBIGUITY_ONLY_SPLITS:
                reviews.append(CompositeReview(legacy_requirement_index=index,
                    old_requirement=old, decision="split", resulting_proposition_ids=tuple(child_ids),
                    rationale=("The components are independently meaningful; equal fractional weights "
                               "preserve the old requirement's total weight.")))
        else:
            prop_id = _slug(prefix, task.task_id, index)
            ids_by_legacy[(index, "")] = prop_id
            requirement_type = (RequirementType.COHESIVE_COMPOSITE
                if key in _RETAINED_COMPOSITES else RequirementType.ATOMIC)
            description = _DESCRIPTION_OVERRIDES.get(key, old.rstrip(".") + ".")
            pending.append((index, "", description, RationalWeight(numerator=1, denominator=1),
                            requirement_type))
            if key in _RETAINED_COMPOSITES:
                reviews.append(CompositeReview(legacy_requirement_index=index,
                    old_requirement=old, decision="retain", resulting_proposition_ids=(prop_id,),
                    rationale="The affected lots and sensor-fault context form one recall event."))
        if key in _AMBIGUITIES:
            problem, resolution = _AMBIGUITIES[key]
            resulting = tuple(item.proposition_id for item in propositions
                              if item.legacy_requirement_index == index)
            # Pending IDs are used because propositions are assembled after dependency resolution.
            resulting = tuple(ids_by_legacy[k] for k in ids_by_legacy if k[0] == index)
            ambiguity.append(AmbiguityResolution(old_requirement=old, ambiguity=problem,
                resolution=resolution, new_proposition_ids=resulting,
                rationale="The resolution makes the benchmark-owned information boundary explicit."))
    for index, suffix, description, weight, requirement_type in pending:
        prop_id = ids_by_legacy[(index, suffix)]
        template = _DERIVATION_TEMPLATES.get((family, task.task_id, index, suffix))
        derivations: tuple[DerivationRule, ...] = ()
        equivalences: tuple[DirectionalEquivalence, ...] = ()
        if template:
            policy, expression, dependencies, rationale = template
            dependency_ids = tuple(ids_by_legacy[item] for item in dependencies)
            derivations = (DerivationRule(
                derivation_id=f"derive-{prop_id}", policy=policy, expression=expression,
                requires_proposition_ids=dependency_ids, rationale=rationale),)
            if policy is DerivationPolicy.ALLOWED:
                equivalences = (DirectionalEquivalence(alternative=expression,
                    preserves_information=rationale),)
        propositions.append(SemanticProposition(
            proposition_id=prop_id, legacy_requirement_index=index,
            description=description, requirement_type=requirement_type, weight=weight,
            source_evidence=SourceEvidence(quote=_evidence(source, description)),
            deterministic_anchors=_anchors(description), accepted_equivalences=equivalences,
            derivations=derivations,
            notes=("Interpretive wording should return AMBIGUOUS when support and materiality "
                   "cannot be determined from this evidence.")))
    forbidden = tuple(str(item) for item in
                      task.evaluation_metadata.get("deterministic_constraints", {}).get(
                          "forbidden_strings", []))
    applies = ((f"routing-v1.2:{task.task_id}",) if family == "routing" else
               (f"foundation-v2:{task.task_id}", f"foundation-v3:{task.task_id}"))
    return SummarizationTaskSpecification(
        specification_id=f"{prefix}-{task.task_id}", canonical_task_id=task.task_id,
        applies_to=applies, source_text=source, legacy_requirements=requirements,
        propositions=tuple(propositions), composite_reviews=tuple(reviews),
        ambiguity_resolutions=tuple(ambiguity), material_semantic_traps=forbidden)


def build_specification() -> PropositionSpecification:
    routing = load_dataset(ROUTING_V11)
    foundation_v2 = load_dataset(FOUNDATION_V2)
    foundation_v3 = load_dataset(FOUNDATION_V3)
    v2 = {task.task_id: task for task in foundation_v2.tasks if task.category == "summarization"}
    v3 = {task.task_id: task for task in foundation_v3.tasks if task.category == "summarization"}
    def semantic_identity(task: BenchmarkTask):
        return task.prompt, task.evaluation_metadata, task.acceptable_threshold
    if {key: semantic_identity(value) for key, value in v2.items()} != {
            key: semantic_identity(value) for key, value in v3.items()}:
        raise ValueError("Foundation V2/V3 summarization definitions are no longer canonical aliases")
    tasks = tuple(_task_spec(task, "rb12") for task in routing.tasks
                  if task.category == "summarization") + tuple(
        _task_spec(task, "foundation-shared") for task in v2.values())
    return PropositionSpecification(
        source_benchmarks=("routellm-routing-benchmark-v1:1.1.0",
                           "routellm-foundation-v2:2.0.0",
                           "routellm-foundation-v3:3.0.0"), tasks=tasks)


def _validation_cases() -> tuple[dict[str, str], ...]:
    return (
        {"phenomenon":"DIRECT_ENTAILMENT","label":"ENTAILED","source":"The archive opens at 08:30 on Tuesday.","proposition":"The archive opens Tuesday at 08:30.","candidate":"The archive will open Tuesday at 8:30 a.m."},
        {"phenomenon":"DIRECT_ENTAILMENT","label":"ENTAILED","source":"Mara approved the revised budget.","proposition":"Mara approved the revised budget.","candidate":"Mara approved the revised budget."},
        {"phenomenon":"LEXICAL_PARAPHRASE","label":"ENTAILED","source":"The board postponed the vote until June.","proposition":"The vote was postponed until June.","candidate":"The board delayed its vote to June."},
        {"phenomenon":"LEXICAL_PARAPHRASE","label":"ENTAILED","source":"Rail service was unaffected by the inspection.","proposition":"Rail service was unaffected.","candidate":"The inspection did not disrupt trains."},
        {"phenomenon":"SEMANTIC_COMPRESSION","label":"ENTAILED","source":"Northbound trains stopped at 08:10 and resumed at 08:25.","proposition":"Northbound service was interrupted.","candidate":"Northbound service briefly paused."},
        {"phenomenon":"SEMANTIC_COMPRESSION","label":"ENTAILED","source":"The clinic received no deliveries for three days and canceled appointments.","proposition":"The delivery interruption affected appointments.","candidate":"A three-day delivery gap disrupted appointments."},
        {"phenomenon":"SEMANTIC_COMPRESSION","label":"ENTAILED","source":"The same invoice was entered twice; staff removed the second entry.","proposition":"Staff removed the duplicate invoice entry.","candidate":"Staff deleted the duplicated invoice record."},
        {"phenomenon":"VALID_DERIVATION","label":"ENTAILED","source":"Of 200 registered participants, 150 attended.","proposition":"Three quarters of registered participants attended.","candidate":"Attendance was 75% of registrations."},
        {"phenomenon":"VALID_DERIVATION","label":"ENTAILED","source":"The price fell from $20 to $15.","proposition":"The price decreased by $5.","candidate":"The item became five dollars cheaper."},
        {"phenomenon":"VALID_DERIVATION","label":"ENTAILED","source":"The outage began at 11:05 and ended at 11:35.","proposition":"The outage lasted 30 minutes.","candidate":"Service was down for half an hour."},
        {"phenomenon":"INVALID_DERIVATION","label":"NOT_ENTAILED","source":"The price fell from $20 to $15.","proposition":"The final price was $15.","candidate":"The price decreased by $5."},
        {"phenomenon":"INVALID_DERIVATION","label":"NOT_ENTAILED","source":"Three of six samples passed.","proposition":"Exactly three samples passed.","candidate":"Half of the samples passed."},
        {"phenomenon":"INVALID_DERIVATION","label":"NOT_ENTAILED","source":"Restrictions applied Monday through Wednesday.","proposition":"Restrictions ended Wednesday.","candidate":"Restrictions lasted three days."},
        {"phenomenon":"MISSING_FACT","label":"NOT_ENTAILED","source":"The permit was approved Thursday after a diagram arrived Wednesday.","proposition":"The diagram arrived Wednesday.","candidate":"The permit was approved Thursday."},
        {"phenomenon":"MISSING_FACT","label":"NOT_ENTAILED","source":"The shipment contained rice and medical supplies.","proposition":"The shipment contained medical supplies.","candidate":"The shipment contained rice."},
        {"phenomenon":"CONTRADICTION","label":"NOT_ENTAILED","source":"The east entrance remained closed throughout the event.","proposition":"The east entrance remained closed.","candidate":"The east entrance stayed open during the event."},
        {"phenomenon":"CONTRADICTION","label":"NOT_ENTAILED","source":"No customer records were lost.","proposition":"No customer records were lost.","candidate":"Some customer records were lost."},
        {"phenomenon":"UNSUPPORTED_CAUSATION","label":"NOT_ENTAILED","source":"A patch was installed Monday. Errors declined Tuesday.","proposition":"The patch caused errors to decline.","candidate":"Monday's patch caused Tuesday's decline in errors."},
        {"phenomenon":"UNSUPPORTED_CAUSATION","label":"NOT_ENTAILED","source":"The alarm sounded before staff evacuated.","proposition":"The alarm caused the evacuation.","candidate":"Staff evacuated because the alarm sounded."},
        {"phenomenon":"UNSUPPORTED_CAUSATION","label":"AMBIGUOUS","source":"A fan stopped and the server overheated moments later.","proposition":"The stopped fan caused the overheating.","candidate":"The fan failure led to overheating."},
        {"phenomenon":"ATTRIBUTION_CHANGE","label":"NOT_ENTAILED","source":"Mina proposed the amendment; the committee approved it.","proposition":"Mina proposed the amendment.","candidate":"The committee proposed the amendment."},
        {"phenomenon":"ATTRIBUTION_CHANGE","label":"AMBIGUOUS","source":"The laboratory issued the report after Ana completed the analysis.","proposition":"Ana authored the report.","candidate":"Ana's report was issued by the laboratory."},
        {"phenomenon":"CERTAINTY_CHANGE","label":"NOT_ENTAILED","source":"Analysts said demand may rise.","proposition":"Analysts expressed uncertainty about rising demand.","candidate":"Analysts said demand will rise."},
        {"phenomenon":"CERTAINTY_CHANGE","label":"AMBIGUOUS","source":"The pilot appears successful on the available measures.","proposition":"The pilot was successful.","candidate":"The successful pilot met the available measures."},
    )


def build_validation_set() -> tuple[tuple[PropositionJudgeInput, ...],
                                    dict[str, ValidationExpectation]]:
    inputs = []
    labels = {}
    for index, case in enumerate(_validation_cases(), 1):
        case_id = f"prop-live-{index:02d}"
        inputs.append(PropositionJudgeInput(case_id=case_id,
            source_evidence=case["source"], required_proposition=case["proposition"],
            candidate_text=case["candidate"]))
        labels[case_id] = ValidationExpectation(
            expected_verdict=EntailmentVerdict(case["label"]),
            phenomenon=case["phenomenon"],
            rationale="Frozen human-authored boundary decision for proposition-level validation.")
    return tuple(inputs), labels


def build_routing_v12(specification: PropositionSpecification) -> BenchmarkDataset:
    old = load_dataset(ROUTING_V11)
    by_task = {item.canonical_task_id: item for item in specification.tasks
               if item.specification_id.startswith("rb12-")}
    tasks = []
    for task in old.tasks:
        if task.category != "summarization":
            tasks.append(task)
            continue
        spec = by_task[task.task_id]
        metadata = dict(task.evaluation_metadata)
        metadata["legacy_semantic_requirements"] = metadata["semantic_requirements"]
        metadata["semantic_requirements"] = [item.description for item in spec.propositions]
        metadata["semantic_specification_version"] = SPECIFICATION_VERSION
        metadata["semantic_specification_id"] = spec.specification_id
        metadata["weighted_propositions"] = [{
            "proposition_id": item.proposition_id,
            "weight": f"{item.weight.numerator}/{item.weight.denominator}",
        } for item in spec.propositions]
        tasks.append(task.model_copy(update={"evaluation_metadata": metadata}))
    return old.model_copy(update={"version": ROUTING_DATASET_VERSION, "tasks": tuple(tasks)})


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _review_markdown(spec: PropositionSpecification, new_dataset_sha: str,
                     validation_inputs: tuple[PropositionJudgeInput, ...],
                     labels: dict[str, ValidationExpectation]) -> str:
    lines = [
        "# Phase 9 summarization proposition specification — human review",
        "", f"- Specification: {SPECIFICATION_VERSION}",
        f"- Proposed Routing Benchmark dataset: {ROUTING_DATASET_VERSION}",
        f"- Proposed dataset SHA-256: `{new_dataset_sha}`",
        "- Implemented evaluator: unchanged at historical 1.2.0",
        "- Status: human review required; no paid validation authorized", "",
        "## Coverage rule", "",
        "Coverage is the exact sum of ENTAILED proposition weights divided by total weight. "
        "Any AMBIGUOUS verdict makes the evaluation incomplete. Split weights preserve each "
        "legacy requirement's total weight. The existing 0.8 threshold is recommended for "
        "human approval because it represents at least 80% of benchmark-owned semantic weight; "
        "it is not selected or implemented by this specification.", "",
        "## Material-error contract", "",
        "Every finding requires the candidate claim, specific source evidence, one frozen error "
        "category, materiality (`material` or `borderline`), and a concise reason. Categories: "
        + ", ".join(item.value for item in MaterialErrorCategory) + ".", "",
        "Clearly supported characterizations are not errors. Clearly unsupported substantive "
        "assertions are material errors. Genuinely borderline interpretive wording produces "
        "AMBIGUOUS and requires review; it is not an automatic veto.", "",
    ]
    for task in spec.tasks:
        lines += [f"## {task.specification_id}", "", f"**Applies to:** {', '.join(task.applies_to)}",
                  "", f"**Source:** {task.source_text}", "", "| Proposition | Weight | Evidence | Anchors | Derivation |",
                  "|---|---:|---|---|---|"]
        for item in task.propositions:
            anchors = ", ".join(anchor.value for anchor in item.deterministic_anchors) or "—"
            derivation = "; ".join(f"{rule.policy}: {rule.expression}" for rule in item.derivations) or "—"
            lines.append(f"| `{item.proposition_id}` — {item.description} | "
                         f"{item.weight.numerator}/{item.weight.denominator} | "
                         f"{item.source_evidence.quote} | {anchors} | {derivation} |")
        total = sum((item.weight.fraction for item in task.propositions), Fraction())
        min_weight = min(item.weight.fraction for item in task.propositions)
        lines += ["", f"**Total weight:** {total}; **minimum meaningful omission:** {min_weight}; "
                  "the table below audits every single-proposition omission.", "",
                  "| Omitted proposition | Omitted weight | Resulting coverage |",
                  "|---|---:|---:|"]
        for item in task.propositions:
            lines.append(f"| `{item.proposition_id}` | {item.weight.fraction} | "
                         f"{(total - item.weight.fraction) / total} |")
        for review in task.composite_reviews:
            lines += [f"- Composite review: **{review.decision.upper()}** — {review.old_requirement} — {review.rationale}"]
        for resolution in task.ambiguity_resolutions:
            lines += [f"- Ambiguity — old: {resolution.old_requirement}",
                      f"  - Problem: {resolution.ambiguity}",
                      f"  - Resolution: {resolution.resolution}",
                      f"  - New propositions: {', '.join(resolution.new_proposition_ids)}",
                      f"  - Rationale: {resolution.rationale}"]
        if task.material_semantic_traps:
            lines += [f"- Material traps: {', '.join(task.material_semantic_traps)}"]
        lines.append("")
    lines += ["## Frozen 24-case validation set", "",
              "Expected labels are stored separately from judge inputs.", "",
              "| Case | Phenomenon | Expected | Source | Proposition | Candidate |",
              "|---|---|---|---|---|---|"]
    for item in validation_inputs:
        expected = labels[item.case_id]
        lines.append(f"| {item.case_id} | {expected.phenomenon} | {expected.expected_verdict} | "
                     f"{item.source_evidence} | {item.required_proposition} | {item.candidate_text} |")
    lines += ["", "Historical Astra 1.1 and 1.2 expected labels remain unchanged. The earlier "
              "descriptive-characterization and duration-compression expectations are annotated "
              "as underspecified historical fixtures; they are not relabeled here.", ""]
    return "\n".join(lines)


def write_specification_artifacts() -> dict[str, JsonValue]:
    specification = build_specification()
    inputs, labels = build_validation_set()
    new_dataset = build_routing_v12(specification)
    spec_bytes = _json_bytes(specification.model_dump(mode="json"))
    input_bytes = _json_bytes({"validation_set_version": VALIDATION_SET_VERSION,
                               "judge_inputs": [item.model_dump(mode="json") for item in inputs]})
    label_bytes = _json_bytes({"validation_set_version": VALIDATION_SET_VERSION,
                               "expected_labels": {key: value.model_dump(mode="json")
                                                   for key, value in labels.items()}})
    dataset_bytes = _json_bytes(new_dataset.model_dump(mode="json"))
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    review = _review_markdown(specification, dataset_sha, inputs, labels).encode()
    files = {SPEC_PATH: spec_bytes, VALIDATION_INPUT_PATH: input_bytes,
             VALIDATION_LABEL_PATH: label_bytes, ROUTING_V12: dataset_bytes,
             REVIEW_PATH: review}
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {"specification_version": SPECIFICATION_VERSION,
            "dataset_version": ROUTING_DATASET_VERSION,
            "dataset_sha256": dataset_sha,
            "specification_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "validation_inputs_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "validation_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
            "task_specifications": len(specification.tasks),
            "propositions": sum(len(item.propositions) for item in specification.tasks),
            "validation_cases": len(inputs)}
