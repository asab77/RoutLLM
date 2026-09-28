"""Strict offline execution-readiness evidence for paid benchmark protocols."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.models.schemas import DomainModel
from adaptive_llm_gateway.providers.gateway_config import JUDGE_SELECTION_MODELS

from .models import load_dataset
from .routing_benchmark_v1 import (
    ROUTING_V12_PROTOCOL_NAME,
    ROUTING_V12_CORRECTED_PROTOCOL_VERSION,
    ROUTING_V12_MINIMAL_REASONING_PROTOCOL_VERSION,
    ROUTING_V12_NATIVE_MINIMAL_REASONING_PROTOCOL_VERSION,
    SUPPORTED_EXECUTION_PROTOCOLS,
    load_execution_protocol,
)

PRICING_READINESS_VERSION = "1.0.0"
READY_STATUS = "READY_FOR_PAID_EXECUTION"


class ModelPricingEvidence(DomainModel):
    model_id: str = Field(min_length=1)
    upstream_model_slug: str = Field(min_length=1)
    input_cost_per_1m_tokens: Decimal = Field(ge=0)
    output_cost_per_1m_tokens: Decimal = Field(ge=0)
    verification: Literal["HUMAN_WEB_VERIFIED", "REPOSITORY_CATALOG_VERIFIED"]
    verified_on: date
    evidence: tuple[str, ...] = Field(min_length=1)


class PricingReadinessRecord(DomainModel):
    version: str = PRICING_READINESS_VERSION
    status: Literal["READY_FOR_PAID_EXECUTION", "REQUIRES_REVERIFICATION"]
    protocol: str
    protocol_version: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recorded_on: date
    models: tuple[ModelPricingEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_models(self):
        ids = [item.model_id for item in self.models]
        slugs = [item.upstream_model_slug for item in self.models]
        if len(ids) != len(set(ids)) or len(slugs) != len(set(slugs)):
            raise ValueError("pricing evidence model identities must be unique")
        return self


def load_pricing_readiness(path: Path, *, protocol: dict,
                           protocol_sha256: str,
                           candidate_models: tuple[ModelConfig, ...]) -> PricingReadinessRecord:
    record = PricingReadinessRecord.model_validate_json(path.read_bytes())
    identity = (protocol.get("protocol"), protocol.get("version"))
    contract = SUPPORTED_EXECUTION_PROTOCOLS.get(identity)
    if contract is None:
        raise ValueError("Pricing evidence references an unsupported protocol")
    if record.status != READY_STATUS:
        raise ValueError("Protocol pricing is not ready for paid execution")
    if (record.protocol, record.protocol_version, record.protocol_sha256) != (
            contract.protocol, contract.version, protocol_sha256):
        raise ValueError("Pricing evidence protocol identity mismatch")
    astra = next(model for model in JUDGE_SELECTION_MODELS
                 if model.model_id == "judge-gpt-6-astra")
    expected = {item.model_id: item for item in (*candidate_models, astra)}
    actual = {item.model_id: item for item in record.models}
    if set(actual) != set(expected):
        raise ValueError("Pricing evidence must cover every paid model exactly once")
    for model_id, model in expected.items():
        evidence = actual[model_id]
        if (evidence.upstream_model_slug,
                evidence.input_cost_per_1m_tokens,
                evidence.output_cost_per_1m_tokens) != (
                model.provider_model_name,
                model.input_cost_per_1m_tokens,
                model.output_cost_per_1m_tokens):
            raise ValueError(f"Pricing evidence conflicts with {model_id}")
    return record


def corrected_pilot_dry_run(*, protocol_path: Path, readiness_path: Path,
                            dataset_path: Path, split_path: Path,
                            evaluator_path: Path, specification_path: Path,
                            pilot_path: Path, cost_path: Path) -> dict:
    protocol, candidates, protocol_sha = load_execution_protocol(protocol_path)
    identity = (protocol["protocol"], protocol["version"])
    contract = SUPPORTED_EXECUTION_PROTOCOLS[identity]
    if not contract.requires_pricing_readiness:
        raise ValueError("Corrected pilot requires the Protocol 1.4 readiness contract")
    readiness = load_pricing_readiness(
        readiness_path, protocol=protocol, protocol_sha256=protocol_sha,
        candidate_models=candidates)
    dataset = load_dataset(dataset_path)
    if dataset.sha256 != contract.dataset_sha256:
        raise ValueError("Corrected pilot dataset hash mismatch")
    split_digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_digest != protocol["split_manifest_sha256"]:
        raise ValueError("Corrected pilot split manifest hash mismatch")
    evaluator_digest = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    if evaluator_digest != protocol["evaluator_manifest_sha256"]:
        raise ValueError("Corrected pilot evaluator manifest hash mismatch")
    evaluator = json.loads(evaluator_path.read_bytes())
    if evaluator.get("version") != "1.3.0":
        raise ValueError("Corrected pilot requires evaluator 1.3")
    specification_digest = hashlib.sha256(specification_path.read_bytes()).hexdigest()
    if specification_digest != protocol["summarization_specification_sha256"]:
        raise ValueError("Corrected pilot specification hash mismatch")
    split = json.loads(split_path.read_bytes())
    pilot = json.loads(pilot_path.read_bytes())
    split_by_task = {item["task_id"]: item for item in split["entries"]}
    tasks = pilot["tasks"]
    if len(tasks) != 21 or len({item["task_id"] for item in tasks}) != 21:
        raise ValueError("Corrected pilot must contain 21 distinct tasks")
    if any(split_by_task[item["task_id"]]["split"] != "train" for item in tasks):
        raise ValueError("Corrected pilot must be TRAIN-only")
    dataset_by_task = {task.task_id: task for task in dataset.tasks}
    if any(item["task_id"] not in dataset_by_task for item in tasks):
        raise ValueError("Corrected pilot contains an unknown task")
    allowances = {
        item["task_id"]: {
            model.model_id: model.provider_output_allowance(
                dataset_by_task[item["task_id"]].max_output_tokens,
                category=dataset_by_task[item["task_id"]].category,
            ) for model in candidates
        } for item in tasks
    }
    costs = json.loads(cost_path.read_bytes())["corrected_pilot"]
    total = Decimal(costs["expected_total_cost_usd"])
    worst_case = Decimal(costs["worst_case_cost_usd"])
    ceiling = Decimal("0.60")
    if total > ceiling or worst_case > ceiling:
        raise ValueError("Corrected pilot projected cost exceeds its ceiling")
    return {
        "status": READY_STATUS,
        "protocol": protocol["protocol"],
        "protocol_version": protocol["version"],
        "protocol_sha256": protocol_sha,
        "pricing_readiness_version": readiness.version,
        "pricing_readiness_sha256": hashlib.sha256(readiness_path.read_bytes()).hexdigest(),
        "dataset_sha256": dataset.sha256,
        "split_manifest_sha256": split_digest,
        "specification_sha256": specification_digest,
        "evaluator_version": evaluator["version"],
        "evaluator_manifest_sha256": evaluator_digest,
        "pilot_tasks": len(tasks),
        "train_tasks": len(tasks),
        "development_tasks": 0,
        "final_tasks": 0,
        "candidate_count": len(candidates),
        "maximum_candidate_attempts": len(tasks) * len(candidates),
        "maximum_astra_calls": sum(item["internal_category"] == "summarization"
                                   for item in tasks) * len(candidates),
        "effective_max_output_tokens": allowances,
        "projected_candidate_cost_usd": costs["candidate_cost_usd"],
        "projected_astra_cost_usd": costs["expected_cost_usd"],
        "projected_total_cost_usd": costs["expected_total_cost_usd"],
        "projected_worst_case_cost_usd": costs["worst_case_cost_usd"],
        "authorized_ceiling_usd": str(ceiling),
        "provider_calls_made": 0,
    }


def gemini_confirmation_dry_run(*, protocol_path: Path, readiness_path: Path,
                                dataset_path: Path, split_path: Path,
                                confirmation_path: Path) -> dict:
    """Validate the five-case Gemini confirmation without crossing its paid gate."""
    protocol, candidates, protocol_sha = load_execution_protocol(protocol_path)
    if (protocol["protocol"], protocol["version"]) not in {
            (ROUTING_V12_PROTOCOL_NAME, ROUTING_V12_CORRECTED_PROTOCOL_VERSION),
            (ROUTING_V12_PROTOCOL_NAME, ROUTING_V12_MINIMAL_REASONING_PROTOCOL_VERSION),
            (ROUTING_V12_PROTOCOL_NAME, ROUTING_V12_NATIVE_MINIMAL_REASONING_PROTOCOL_VERSION),
    }:
        raise ValueError("Gemini confirmation requires Protocol 1.5, 1.6, or 1.7")
    readiness = load_pricing_readiness(
        readiness_path, protocol=protocol, protocol_sha256=protocol_sha,
        candidate_models=candidates)
    dataset = load_dataset(dataset_path)
    contract = SUPPORTED_EXECUTION_PROTOCOLS[(protocol["protocol"], protocol["version"])]
    if dataset.sha256 != contract.dataset_sha256:
        raise ValueError("Gemini confirmation dataset hash mismatch")
    split_digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_digest != protocol["split_manifest_sha256"]:
        raise ValueError("Gemini confirmation split hash mismatch")
    split = json.loads(split_path.read_bytes())
    train_ids = {item["task_id"] for item in split["entries"] if item["split"] == "train"}
    manifest = json.loads(confirmation_path.read_bytes())
    expected_ids = {
        "coding-easy-001", "coding-medium-001", "coding-hard-001",
        "summarization-easy-001", "summarization-medium-001",
    }
    cases = manifest.get("cases")
    if not isinstance(cases, list) or {item.get("task_id") for item in cases} != expected_ids:
        raise ValueError("Gemini confirmation must contain exactly the five failed cases")
    gemini = next(model for model in candidates
                  if model.model_id == "candidate-gemini-3-flash")
    if (manifest.get("protocol_sha256") != protocol_sha
            or manifest.get("execution_readiness_sha256") != hashlib.sha256(
                readiness_path.read_bytes()).hexdigest()
            or manifest.get("candidate_id") != gemini.model_id
            or manifest.get("reasoning_effort") != gemini.reasoning_effort.value
            or manifest.get("reasoning_control", "gateway_shared")
            != gemini.reasoning_control.value
            or manifest.get("maximum_candidate_calls") != 5
            or manifest.get("candidate_retries") != 0
            or manifest.get("astra_calls") != 0
            or manifest.get("quality_evaluation") is not False):
        raise ValueError("Gemini confirmation execution contract mismatch")
    task_by_id = {task.task_id: task for task in dataset.tasks}
    if not expected_ids <= train_ids:
        raise ValueError("Gemini confirmation cases must remain TRAIN-only")
    for item in cases:
        task = task_by_id[item["task_id"]]
        expected_allowance = gemini.provider_output_allowance(
            task.max_output_tokens, category=task.category)
        if (item.get("visible_output_requirement") != task.max_output_tokens
                or item.get("provider_output_allowance") != expected_allowance):
            raise ValueError("Gemini confirmation allowance mismatch")
    report = {
        "status": "READY_FOR_CONFIRMATION",
        "protocol_sha256": protocol_sha,
        "pricing_readiness_version": readiness.version,
        "case_ids": sorted(expected_ids),
        "candidate_id": gemini.model_id,
        "reasoning_effort": gemini.reasoning_effort.value,
        "reasoning_reserve_tokens": gemini.output_token_policy.reasoning_headroom_tokens,
        "maximum_candidate_calls": 5,
        "astra_calls": 0,
        "provider_calls_made": 0,
    }
    if protocol["version"] == ROUTING_V12_NATIVE_MINIMAL_REASONING_PROTOCOL_VERSION:
        report["reasoning_control"] = gemini.reasoning_control.value
    return report
