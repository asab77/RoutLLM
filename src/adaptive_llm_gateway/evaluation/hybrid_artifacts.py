"""Deterministic proposed manifests for the hybrid semantic evaluator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adaptive_llm_gateway.benchmarks.models import load_dataset
from adaptive_llm_gateway.benchmarks.summarization_spec import PropositionSpecification

from .hybrid_judge import (
    HYBRID_JUDGE_PROMPT_VERSION,
    HYBRID_SEMANTIC_EVALUATOR_VERSION,
)

PROPOSED_PROTOCOL_VERSION = "1.4.0"
PROPOSED_ROOT = Path("benchmarks/protocols/routing-benchmark-v1.2")
IMPLEMENTATION_FILES = (
    Path("src/adaptive_llm_gateway/evaluation/hybrid_judge.py"),
    Path("src/adaptive_llm_gateway/evaluation/service.py"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write_proposed_manifests(*, specification_path: Path,
                             dataset_path: Path,
                             historical_evaluator_path: Path,
                             historical_protocol_path: Path,
                             cost_plan: dict) -> dict[str, str]:
    specification = PropositionSpecification.model_validate_json(
        specification_path.read_bytes())
    spec_by_task = {item.canonical_task_id: item for item in specification.tasks
                    if item.specification_id.startswith("rb12-")}
    old_evaluator = json.loads(historical_evaluator_path.read_text())
    entries = []
    for entry in old_evaluator["entries"]:
        updated = dict(entry)
        if entry["category"] == "summarization":
            task_spec = spec_by_task[entry["task_id"]]
            updated |= {
                "evaluator_type": "hybrid_proposition_semantic_summary",
                "judge_prompt_version": HYBRID_JUDGE_PROMPT_VERSION,
                "rubric_or_schema_identity": task_spec.specification_id,
                "semantic_judge_version": HYBRID_SEMANTIC_EVALUATOR_VERSION,
            }
        entries.append(updated)
    evaluator = {
        "benchmark": "routellm-routing-benchmark-v1.2",
        "version": HYBRID_SEMANTIC_EVALUATOR_VERSION,
        "implementation_files": {str(path): sha256(path) for path in IMPLEMENTATION_FILES},
        "judge_call_count_per_summary": 1,
        "judge_prompt_version": HYBRID_JUDGE_PROMPT_VERSION,
        "proposition_specification_sha256": sha256(specification_path),
        "entries": entries,
    }
    PROPOSED_ROOT.mkdir(parents=True, exist_ok=True)
    evaluator_path = PROPOSED_ROOT / "evaluator-manifest.json"
    evaluator_path.write_bytes(canonical_bytes(evaluator))

    historical_protocol = json.loads(historical_protocol_path.read_text())
    dataset = load_dataset(dataset_path)
    protocol = historical_protocol | {
        "protocol": "routellm-routing-benchmark-v1.2",
        "version": PROPOSED_PROTOCOL_VERSION,
        "status": "PROPOSED_AWAITING_LIVE_VALIDATION",
        "paid_execution_authorized": False,
        "dataset_version": dataset.version,
        "dataset_sha256": sha256(dataset_path),
        "summarization_specification_version": specification.specification_version,
        "summarization_specification_sha256": sha256(specification_path),
        "evaluator_manifest_sha256": sha256(evaluator_path),
        "protocol_bump_reason": (
            "Dataset identity and summarization execution semantics changed from a holistic "
            "judge score to benchmark-owned proposition verdicts with deterministic aggregation."),
    }
    protocol_path = PROPOSED_ROOT / "protocol.json"
    protocol_path.write_bytes(canonical_bytes(protocol))
    cost_path = PROPOSED_ROOT / "cost-estimate.json"
    cost_path.write_bytes(canonical_bytes(cost_plan))
    identities = {
        "evaluator_manifest_sha256": sha256(evaluator_path),
        "protocol_sha256": sha256(protocol_path),
        "cost_estimate_sha256": sha256(cost_path),
        "dataset_sha256": sha256(dataset_path),
        "specification_sha256": sha256(specification_path),
    }
    (PROPOSED_ROOT / "identities.json").write_bytes(canonical_bytes(identities))
    return identities
