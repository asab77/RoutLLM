import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.benchmarks.models import load_dataset
from adaptive_llm_gateway.benchmarks.summarization_spec import (
    EntailmentVerdict, MaterialErrorCategory, MaterialErrorFinding,
    PropositionJudgeInput, RequirementType, SourceEvidence,
    SPECIFICATION_VERSION, VALIDATION_INPUT_PATH, VALIDATION_LABEL_PATH,
    DerivationPolicy, build_specification, build_validation_set,
    weighted_coverage,
)
from adaptive_llm_gateway.evaluation.judge import SEMANTIC_JUDGE_VERSION

OLD_DATASET_SHA = "70152d1ac15e827a0ff940cf5becf76701578d0a62d773b4ad6e3823c3fa27ca"
SPLIT_SHA = "98c639be29da4e11fdf48073d74e83805f16fdfb72b0102b5f5543213ab4960b"
PILOT_TREE_SHA = "9c194b025d3acfd6f12a2ce1f0f0f4d673757252665f28fd0b2d518ad815425a"
ASTRA_11_SHA = "50fa94804e9f2104c7fb60804cb22bfc5e2289a89ed773df50ebadef4ea556fe"
ASTRA_12_SHA = "96c43e4967bd7a3dbf438431fc27745746bd89ba217fe2831d00aa25b6f59bfe"
FOUNDATION_V2_SHA = "dcab5e9b6f1b8bc063555637e06db0d32ec51056959a7586f2cc411949adc964"
FOUNDATION_V3_SHA = "64eda8e7388233f645906412a2524982ebfb9c848c42ee24a468ba04c31d16e6"
PHASE7_TREE_SHA = "6f1c57247a7fb941aaaa8862aba581cde476d1d4c98078c6942c37b17d657287"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@pytest.fixture(scope="module")
def specification():
    return build_specification()


@pytest.mark.protected_final_data
def test_specification_uses_canonical_foundation_definitions_and_stable_ids(specification):
    assert specification.specification_version == SPECIFICATION_VERSION == "1.0.0"
    assert len(specification.tasks) == 40
    assert sum(len(task.propositions) for task in specification.tasks) == 154
    ids = [item.proposition_id for task in specification.tasks for item in task.propositions]
    assert len(ids) == len(set(ids))
    foundation = [task for task in specification.tasks
                  if task.specification_id.startswith("foundation-shared-")]
    assert len(foundation) == 8
    assert all(len(task.applies_to) == 2 for task in foundation)


@pytest.mark.protected_final_data
def test_atomicity_reviews_are_explicit_and_splits_preserve_weight(specification):
    reviews = [review for task in specification.tasks for review in task.composite_reviews]
    assert Counter(review.decision for review in reviews) == {"split": 14, "retain": 1}
    retained = [item for task in specification.tasks for item in task.propositions
                if item.requirement_type is RequirementType.COHESIVE_COMPOSITE]
    assert len(retained) == 1
    for task in specification.tasks:
        total = sum((item.weight.fraction for item in task.propositions), Fraction())
        assert total == len(task.legacy_requirements)


@pytest.mark.protected_final_data
def test_weighted_coverage_is_exact_and_ambiguous_is_incomplete(specification):
    task = next(item for item in specification.tasks if len(item.propositions) > 3)
    entailed = {item.proposition_id: EntailmentVerdict.ENTAILED for item in task.propositions}
    complete = weighted_coverage(task, entailed)
    assert complete.status == "complete"
    assert (complete.coverage_numerator, complete.coverage_denominator) == (1, 1)
    first = task.propositions[0].proposition_id
    ambiguous = weighted_coverage(task, entailed | {first: EntailmentVerdict.AMBIGUOUS})
    assert ambiguous.status == "incomplete" and ambiguous.coverage_numerator is None
    assert ambiguous.ambiguous_proposition_ids == (first,)


@pytest.mark.protected_final_data
def test_every_proposition_has_source_evidence_anchors_are_helpers(specification):
    propositions = [item for task in specification.tasks for item in task.propositions]
    assert all(item.source_evidence.quote.strip() for item in propositions)
    anchored = [item for item in propositions if item.deterministic_anchors]
    assert len(anchored) == 113
    assert all(not anchor.required_exactly for item in anchored
               for anchor in item.deterministic_anchors)


@pytest.mark.protected_final_data
def test_equivalence_and_derivation_policies_are_directional_and_complete(specification):
    rules = [rule for task in specification.tasks for item in task.propositions
             for rule in item.derivations]
    assert Counter(rule.policy for rule in rules) == {
        DerivationPolicy.ALLOWED: 2,
        DerivationPolicy.CONDITIONAL: 6,
        DerivationPolicy.NOT_ALLOWED: 2,
    }
    assert all(rule.requires_proposition_ids for rule in rules
               if rule.policy is DerivationPolicy.CONDITIONAL)
    equivalences = [equivalence for task in specification.tasks for item in task.propositions
                    for equivalence in item.accepted_equivalences]
    assert len(equivalences) == 2
    assert {item.direction for item in equivalences} == {"alternative_satisfies_proposition"}


@pytest.mark.protected_final_data
def test_known_ambiguities_are_resolved_without_hiding_history(specification):
    resolutions = [item for task in specification.tasks for item in task.ambiguity_resolutions]
    assert len(resolutions) == 3  # Four versioned entries; Foundation V2/V3 share one definition.
    assert sum(len(task.applies_to) for task in specification.tasks
               for _ in task.ambiguity_resolutions) == 4
    assert all(item.old_requirement and item.ambiguity and item.resolution
               and item.new_proposition_ids and item.rationale for item in resolutions)


def test_material_error_schema_requires_claim_evidence_category_and_materiality():
    finding = MaterialErrorFinding(candidate_claim="The patch caused the decline.",
        source_evidence=SourceEvidence(quote="A patch landed Monday. Errors fell Tuesday."),
        category=MaterialErrorCategory.UNSUPPORTED_CAUSATION,
        materiality="material", reason="Sequence alone does not establish causation.")
    assert finding.materiality == "material"
    borderline = finding.model_copy(update={"materiality": "borderline"})
    assert borderline.materiality == "borderline"
    with pytest.raises(ValidationError):
        MaterialErrorFinding(candidate_claim="claim",
            category=MaterialErrorCategory.CONTRADICTION,
            materiality="material", reason="missing evidence")


@pytest.mark.protected_final_data
def test_validation_set_is_exact_balanced_enough_blind_and_diverse():
    inputs, labels = build_validation_set()
    assert len(inputs) == len(labels) == 24
    counts = Counter(item.expected_verdict for item in labels.values())
    assert counts == {EntailmentVerdict.ENTAILED: 10,
                      EntailmentVerdict.NOT_ENTAILED: 11,
                      EntailmentVerdict.AMBIGUOUS: 3}
    phenomena = Counter(item.phenomenon for item in labels.values())
    required = {"DIRECT_ENTAILMENT", "LEXICAL_PARAPHRASE", "SEMANTIC_COMPRESSION",
                "VALID_DERIVATION", "INVALID_DERIVATION", "MISSING_FACT",
                "CONTRADICTION", "UNSUPPORTED_CAUSATION", "ATTRIBUTION_CHANGE",
                "CERTAINTY_CHANGE"}
    assert set(phenomena) == required and min(phenomena.values()) >= 2
    serialized = json.dumps([item.model_dump(mode="json") for item in inputs]).casefold()
    assert all(value not in serialized for value in (
        "expected", "candidate identity", "model identity", "difficulty",
        "historical outcome", "nemotron", "luna", "gemini", "sonnet", "astra"))
    assert sum(serialized.count(value) for value in
               ("faulty", "misconfigured", "allowing", "40-minute outage")) == 0


@pytest.mark.protected_final_data
def test_written_validation_inputs_and_labels_are_separate():
    inputs = json.loads(VALIDATION_INPUT_PATH.read_text())
    labels = json.loads(VALIDATION_LABEL_PATH.read_text())
    assert len(inputs["judge_inputs"]) == len(labels["expected_labels"]) == 24
    assert "expected" not in json.dumps(inputs).casefold()
    assert set(item["case_id"] for item in inputs["judge_inputs"]) == set(labels["expected_labels"])


@pytest.mark.protected_final_data
def test_new_dataset_changes_semantics_without_mutating_v11_or_split():
    old = load_dataset(Path("benchmarks/datasets/routing-benchmark-v1.json"))
    new = load_dataset(Path("benchmarks/datasets/routing-benchmark-v1.2.json"))
    assert old.version == "1.1.0" and new.version == "1.2.0"
    assert sha("benchmarks/datasets/routing-benchmark-v1.json") == OLD_DATASET_SHA
    assert new.sha256 != old.sha256
    assert [item.task_id for item in old.tasks] == [item.task_id for item in new.tasks]
    assert sha("benchmarks/protocols/routing-benchmark-v1/split-manifest.json") == SPLIT_SHA
    old_other = [item.model_dump(mode="json") for item in old.tasks if item.category != "summarization"]
    new_other = [item.model_dump(mode="json") for item in new.tasks if item.category != "summarization"]
    assert old_other == new_other


@pytest.mark.local_evidence
@pytest.mark.protected_final_data
def test_historical_foundation_phase7_dev_final_and_astra_artifacts_are_immutable():
    assert sha("benchmarks/datasets/foundation-v2.json") == FOUNDATION_V2_SHA
    assert sha("benchmarks/datasets/foundation-v3.json") == FOUNDATION_V3_SHA
    assert tree_sha("benchmark-results/61707aba-5ab2-4c16-8ec3-eed74555d69c/phase-7") == PHASE7_TREE_SHA
    assert tree_sha("artifacts/routing-benchmark-v1/pilot-runs/1aa850f6-0862-4704-8f0b-c246c9d990ec") == PILOT_TREE_SHA
    assert sha("artifacts/routing-benchmark-v1/protocol-correction/astra-1.1-live-validation.json") == ASTRA_11_SHA
    assert sha("artifacts/routing-benchmark-v1/semantic-evaluator-1.2/astra-1.2-live-validation.json") == ASTRA_12_SHA
    assert sha("artifacts/routing-benchmark-v1/development-manifest.json") == "93daa569d7149666328ffad28eb2db5fbd1342700235d47be06bf001fef61c0e"
    assert sha("artifacts/routing-benchmark-v1/final-manifest.json") == "cca8f7f57df723f6f92624394c382544f972165d4087c69d4f3b1e402b43ef65"
    assert SEMANTIC_JUDGE_VERSION == "1.2.0"
