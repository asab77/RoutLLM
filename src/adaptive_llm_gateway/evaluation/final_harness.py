"""Fail-closed, one-time offline FINAL evaluation and reporting harness.

Normal use is preflight-only. Paid candidate collection remains in the benchmark
runner; paid semantic judging requires the explicit ``semantic-judge --execute``
subcommand and the independently authorized operational record.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pickle
import stat
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator
from sklearn.pipeline import Pipeline

from adaptive_llm_gateway.benchmarks.execution_readiness import READY_STATUS
from adaptive_llm_gateway.benchmarks.features import (
    RequestFeatures,
    feature_extractor_sha256,
    verify_request_feature_binding,
)
from adaptive_llm_gateway.benchmarks.models import BenchmarkResult, BenchmarkRun
from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import load_execution_protocol
from adaptive_llm_gateway.benchmarks.summarization_spec import PropositionSpecification
from adaptive_llm_gateway.evaluation.hybrid_judge import (
    HYBRID_JUDGE_PROMPT_VERSION,
    HYBRID_SEMANTIC_EVALUATOR_VERSION,
    VercelHybridSemanticJudge,
)
from adaptive_llm_gateway.evaluation.aggregation import aggregate
from adaptive_llm_gateway.evaluation.evaluators import evaluator_for
from adaptive_llm_gateway.evaluation.final_ledgers import (
    CandidateAttemptLedger,
    SemanticAttemptLedger,
)
from adaptive_llm_gateway.evaluation.models import EvaluationResult, EvaluationSummary
from adaptive_llm_gateway.evaluation.sandbox import (
    DEFAULT_SANDBOX_IMAGE,
    FUNCTIONAL_EVALUATOR_VERSION,
    DockerPythonSandbox,
)
from adaptive_llm_gateway.evaluation.service import EvaluationService
from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.errors import EvaluationArtifactError
from adaptive_llm_gateway.models.schemas import DomainModel
from adaptive_llm_gateway.pricing import calculate_projected_cost
from adaptive_llm_gateway.providers.gateway_config import (
    GatewaySettings,
    JUDGE_SELECTION_MODELS,
)
from adaptive_llm_gateway.providers.vercel import VercelGatewayProvider
from adaptive_llm_gateway.routing.features import RoutingRequestFeatures
from adaptive_llm_gateway.routing.policy import CandidatePrediction, CostAwareRoutingPolicy
from adaptive_llm_gateway.routing.quality_features import (
    canonical_feature_matrix,
    canonical_from_production,
    resolve_effective_output_allowance,
)


SCHEMA_VERSION = "1.0.0"
POLICY_VERSION = "1.0.0"
THRESHOLD = Decimal("0.80")
EXPECTED_REQUESTS = 42
EXPECTED_CATEGORY_REQUESTS = 6
EXPECTED_CANDIDATE_CALLS = 168
MAXIMUM_JUDGE_CALLS = 24
JUDGE_MODEL_ID = "judge-gpt-6-astra"
JUDGE_UPSTREAM_MODEL = "openai/gpt-6-astra"
JUDGE_MAX_OUTPUT_TOKENS = 768
PRIMARY_BASELINE = "candidate-claude-sonnet-5"
CANDIDATES = (
    "candidate-nemotron-3.5-lightning",
    "candidate-gpt-6-luna",
    "candidate-gemini-3-flash",
    "candidate-claude-sonnet-5",
)
FROZEN_SANDBOX_CONFIGURATION = DockerPythonSandbox().configuration
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_FINAL_ROOT_RELATIVE = Path("artifacts/routing-benchmark-v1/final-runs")
CANONICAL_AUTHORIZATION_RELATIVE = Path(
    "benchmarks/protocols/routing-benchmark-v1.2/final-execution-authorization.json")
BOUND_CONFIGURATION_FILES = (
    "pyproject.toml",
    "requirements.prod.lock",
    "benchmarks/protocols/routing-benchmark-v1.2/final-policy-1.0.json",
    "benchmarks/protocols/routing-benchmark-v1.2/protocol-1.7.json",
    "benchmarks/protocols/routing-benchmark-v1.2/evaluator-manifest.json",
    "benchmarks/protocols/routing-benchmark-v1.2/execution-readiness-1.7.json",
    "benchmarks/specifications/summarization-propositions-v1.0.0.json",
)
EXECUTION_SOURCE_PREFIX = "src/adaptive_llm_gateway/"
SHADOWABLE_EXECUTION_SUFFIXES = frozenset({".py", ".pyi", ".json", ".toml", ".yaml", ".yml"})


def _literal_absolute(path: Path) -> Path:
    """Return an absolute lexical path without following filesystem symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def _verified_repository_root(repository_root: Path) -> Path:
    root = _literal_absolute(repository_root)
    try:
        mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError("FINAL repository root does not exist") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError("FINAL repository root must be a non-symlink directory")
    if root.resolve(strict=True) != root:
        raise ValueError("FINAL repository root must be supplied as its resolved directory")
    return root


def _literal_repository_path(
    repository_root: Path,
    relative: Path,
    *,
    leaf_kind: Literal["directory", "regular-file"],
    require_leaf: bool = False,
) -> Path:
    """Validate a canonical repository-relative path without following symlinks."""
    root = _verified_repository_root(repository_root)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("FINAL canonical path must be repository-relative")
    target = root.joinpath(relative)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("FINAL canonical path escapes the repository root") from exc
    for index in range(len(relative.parts)):
        component = root.joinpath(*relative.parts[:index + 1])
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            if require_leaf:
                raise ValueError("FINAL canonical path does not exist")
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"FINAL canonical path contains a symlink: {component}")
        is_leaf = index == len(relative.parts) - 1
        expected_regular = is_leaf and leaf_kind == "regular-file"
        if expected_regular:
            if not stat.S_ISREG(mode):
                raise ValueError("FINAL authorization must be a regular file")
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"FINAL canonical directory component is not a directory: {component}")
    return target


def canonical_final_root(repository_root: Path = REPOSITORY_ROOT) -> Path:
    return _literal_repository_path(
        repository_root, CANONICAL_FINAL_ROOT_RELATIVE, leaf_kind="directory")


def require_canonical_final_root(
    supplied: Path | None = None,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    expected = canonical_final_root(repository_root)
    if supplied is not None and _literal_absolute(supplied) != expected:
        raise ValueError("FINAL root differs from the canonical experiment root")
    return expected


def ensure_canonical_final_root(repository_root: Path = REPOSITORY_ROOT) -> Path:
    root = canonical_final_root(repository_root)
    root.mkdir(parents=True, exist_ok=True)
    return _literal_repository_path(
        repository_root, CANONICAL_FINAL_ROOT_RELATIVE,
        leaf_kind="directory", require_leaf=True)


def canonical_authorization_path(repository_root: Path = REPOSITORY_ROOT) -> Path:
    return _literal_repository_path(
        repository_root, CANONICAL_AUTHORIZATION_RELATIVE,
        leaf_kind="regular-file")


def require_canonical_authorization_path(
    supplied: Path | None = None,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    expected = _literal_repository_path(
        repository_root, CANONICAL_AUTHORIZATION_RELATIVE,
        leaf_kind="regular-file", require_leaf=True)
    if supplied is not None and _literal_absolute(supplied) != expected:
        raise ValueError("FINAL authorization must use the canonical tracked artifact")
    return expected


def _read_canonical_authorization(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("FINAL authorization must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            value = json.load(source)
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def canonical_publication_directory(repository_root: Path, run_id: UUID) -> Path:
    return canonical_final_root(repository_root) / str(run_id) / "publication"


class FrozenExpectations(DomainModel):
    dataset_sha256: str = "1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005"
    split_sha256: str = "98c639be29da4e11fdf48073d74e83805f16fdfb72b0102b5f5543213ab4960b"
    proposition_sha256: str = "b5568b26d7fdce018e3cfef9020780645e12e02f488ab495b14a1e621dd62c1e"
    evaluator_sha256: str = "7dfb5c29426520edcb2c64491c1d5f62fddfc3bdbb3869599378da781f35d413"
    protocol_sha256: str = "b4cde3954a8ccd1b54684dcb62da303e7bc806248306464ae536b9ff717a0bd8"
    final_manifest_sha256: str = "cca8f7f57df723f6f92624394c382544f972165d4087c69d4f3b1e402b43ef65"
    predictor_sha256: str = "502db83a54c4072c9741a8e4c406498ad88ec97e03bce1ddaf3e0a1b0001c0aa"
    policy_sha256: str = "79da97dca54ffdff254122c7c0345a2136ebd7178cd18f8504757f4d0e31a8ba"
    candidates: tuple[str, ...] = CANDIDATES
    threshold: Decimal = THRESHOLD
    primary_baseline: str = PRIMARY_BASELINE


class FinalPaths(DomainModel):
    dataset: Path = REPOSITORY_ROOT / "benchmarks/datasets/routing-benchmark-v1.2.json"
    split: Path = REPOSITORY_ROOT / "benchmarks/protocols/routing-benchmark-v1/split-manifest.json"
    proposition: Path = REPOSITORY_ROOT / "benchmarks/specifications/summarization-propositions-v1.0.0.json"
    evaluator: Path = REPOSITORY_ROOT / "benchmarks/protocols/routing-benchmark-v1.2/evaluator-manifest.json"
    protocol: Path = REPOSITORY_ROOT / "benchmarks/protocols/routing-benchmark-v1.2/protocol-1.7.json"
    final_manifest: Path = REPOSITORY_ROOT / "artifacts/routing-benchmark-v1/final-manifest.json"
    predictor_directory: Path = REPOSITORY_ROOT / "artifacts/routing-quality/rb12-train-candidate-v1"
    policy: Path = REPOSITORY_ROOT / "benchmarks/protocols/routing-benchmark-v1.2/final-policy-1.0.json"
    pricing_readiness: Path = REPOSITORY_ROOT / (
        "benchmarks/protocols/routing-benchmark-v1.2/execution-readiness-1.7.json")
    authorization: Path = REPOSITORY_ROOT / CANONICAL_AUTHORIZATION_RELATIVE
    confirmation_status: Path = REPOSITORY_ROOT / (
        "artifacts/routing-benchmark-v1/gemini-native-minimal-confirmations/"
        "8f8e90b3-2ca1-4f3d-a181-8cac7aca06f6/status.json")


class FinalPolicy(DomainModel):
    policy_id: str
    policy_version: str
    quality_threshold: Decimal
    candidate_portfolio: tuple[str, ...]
    predictor: dict[str, str]
    selection: dict[str, JsonValue]
    fallback: dict[str, JsonValue]
    implementation_files: dict[str, str]
    serving_cost: dict[str, bool]
    primary_fixed_baseline: str


class FreezeVerification(DomainModel):
    status: Literal["READY", "BLOCKED"]
    experiment_identity: str
    identities: dict[str, str]
    policy_sha256: str
    authorization_status: str
    explicit_authorization: bool
    duplicate_completed_run: bool
    git_commit: str | None = None
    source_hashes: dict[str, str] = Field(default_factory=dict)
    source_set_sha256: str | None = None
    provider_calls_made: int = 0


class CandidateObservation(DomainModel):
    task_id: str
    category: str
    model_id: str
    request_features: dict[str, JsonValue]
    provider_success: bool
    evaluation_status: str
    acceptable: bool | None
    realized_cost_usd: Decimal | None
    judge_failure: bool = False


class RouterSelection(DomainModel):
    task_id: str
    category: str
    selected_model_id: str
    predicted_acceptability: float = Field(ge=0, le=1)
    projected_cost_usd: Decimal = Field(ge=0)
    threshold_satisfied: bool
    fallback_used: bool
    provider_success: bool
    evaluation_status: str
    acceptable: bool | None
    realized_cost_usd: Decimal | None
    judge_failure: bool


class CategoryMetrics(DomainModel):
    category: str
    independent_requests: int
    fully_evaluated: int
    acceptable: int
    task_acceptability: float
    conditional_acceptability: float | None
    provider_failures: int
    judge_failures: int
    unresolved_evaluations: int
    total_realized_inference_cost_usd: Decimal
    mean_realized_cost_per_request_usd: Decimal
    cost_complete: bool


class StrategyMetrics(DomainModel):
    strategy_id: str
    analysis_only: bool = False
    independent_requests: int
    fully_evaluated: int
    acceptable: int
    task_acceptability: float
    conditional_acceptability: float | None
    provider_failures: int
    judge_failures: int
    unresolved_evaluations: int
    total_realized_inference_cost_usd: Decimal
    mean_realized_cost_per_request_usd: Decimal
    cost_complete: bool
    fallback_count: int = 0
    fallback_rate: float | None = None
    categories: tuple[CategoryMetrics, ...]


class PrimaryMetrics(DomainModel):
    baseline: str = PRIMARY_BASELINE
    cost_reduction_percent: float | None
    task_acceptability_percent: float
    independent_requests: int
    cost_complete: bool
    evaluation_complete: bool
    resume_statement: str | None


class CostAccounting(DomainModel):
    serving_cost_definition: str
    candidate_matrix_collection_cost_usd: Decimal
    candidate_matrix_cost_complete: bool
    semantic_judge_cost_usd: Decimal
    excluded_from_primary_cost_comparison: tuple[str, ...]


class FinalResults(DomainModel):
    schema_version: str = SCHEMA_VERSION
    artifact_kind: Literal["offline-held-out-routing-evaluation"] = (
        "offline-held-out-routing-evaluation")
    experiment_identity: str
    identities: dict[str, str]
    git_commit: str
    source_hashes: dict[str, str] = Field(default_factory=dict)
    source_set_sha256: str | None = None
    generated_at: datetime
    threshold: Decimal
    candidate_portfolio: tuple[str, ...]
    primary_baseline: str
    final_request_count: int
    category_counts: dict[str, int]
    candidate_call_count: int
    judge_call_count: int
    strategies: tuple[StrategyMetrics, ...]
    router_selections: tuple[RouterSelection, ...]
    primary_metrics: PrimaryMetrics
    cost_accounting: CostAccounting
    evaluation_complete: bool
    cost_complete: bool
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def primary_baseline_is_predefined(self):
        if self.primary_baseline != PRIMARY_BASELINE:
            raise ValueError("primary baseline must remain fixed Claude Sonnet 5")
        return self


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _tracked_execution_files(repository_root: Path, *, revision: str | None = None) -> tuple[str, ...]:
    if revision is None:
        listing = _git(repository_root, "ls-files", "--", EXECUTION_SOURCE_PREFIX)
    else:
        listing = _git(
            repository_root, "ls-tree", "-r", "--name-only", revision, "--",
            EXECUTION_SOURCE_PREFIX)
    package_files = {line for line in listing.splitlines() if line}
    files = package_files | set(BOUND_CONFIGURATION_FILES)
    files.discard(str(CANONICAL_AUTHORIZATION_RELATIVE))
    return tuple(sorted(files))


def _reject_untracked_execution_files(repository_root: Path, tracked: set[str]) -> None:
    package_root = repository_root / EXECUTION_SOURCE_PREFIX
    for path in package_root.rglob("*"):
        if (not path.is_file() or path.suffix not in SHADOWABLE_EXECUTION_SUFFIXES
                or "__pycache__" in path.parts):
            continue
        relative = path.relative_to(repository_root).as_posix()
        if relative not in tracked:
            raise ValueError(f"untracked execution file could shadow FINAL: {relative}")


def outcome_critical_source_hashes(
    repository_root: Path = REPOSITORY_ROOT,
    sources: tuple[str, ...] | None = None,
) -> dict[str, str]:
    selected = sources or _tracked_execution_files(repository_root)
    hashes = {}
    for relative in selected:
        path = repository_root / relative
        if not path.is_file():
            raise ValueError(f"outcome-critical source is missing: {relative}")
        hashes[relative] = sha256(path)
    return hashes


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repository_root, check=True,
        capture_output=True, text=True)
    return completed.stdout.strip()


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=repository_root, check=True,
        capture_output=True)
    return completed.stdout


def verify_execution_provenance(
    authorization: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    sources: tuple[str, ...] | None = None,
) -> tuple[str, dict[str, str], str]:
    """Require authorized bytes to be exactly the clean committed bytes executing."""
    if _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("FINAL paid execution requires a clean Git worktree")
    execution_commit = _git(repository_root, "rev-parse", "HEAD")
    selected = sources or _tracked_execution_files(repository_root)
    tracked = set(_git(repository_root, "ls-files", "--", *selected).splitlines())
    if tracked != set(selected):
        raise ValueError("outcome-critical source set contains untracked files")
    if sources is None:
        _reject_untracked_execution_files(repository_root, tracked)
    actual_hashes = outcome_critical_source_hashes(repository_root, selected)
    source_set_sha256 = _canonical_digest(actual_hashes)
    approved = authorization.get("implementation")
    if not isinstance(approved, dict):
        raise ValueError("FINAL authorization is not bound to an implementation")
    reviewed_commit = approved.get("git_commit")
    if (not isinstance(reviewed_commit, str) or not reviewed_commit
            or approved.get("source_files") != list(actual_hashes)
            or approved.get("source_hashes") != actual_hashes
            or approved.get("source_set_sha256") != source_set_sha256):
        raise ValueError("executing implementation differs from authorized source identity")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", reviewed_commit, execution_commit],
        cwd=repository_root, capture_output=True)
    if ancestor.returncode != 0:
        raise ValueError("reviewed implementation commit is not an ancestor of execution HEAD")
    try:
        reviewed_sources = sources or _tracked_execution_files(
            repository_root, revision=reviewed_commit)
        if reviewed_sources != selected:
            raise ValueError("outcome-critical source file list changed after authorization")
        reviewed_hashes = {
            source: hashlib.sha256(
                _git_bytes(repository_root, "show", f"{reviewed_commit}:{source}")).hexdigest()
            for source in selected
        }
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "reviewed commit does not contain the complete outcome-critical source set") from exc
    if reviewed_hashes != actual_hashes:
        raise ValueError("outcome-critical source changed after authorization")
    return execution_commit, actual_hashes, source_set_sha256


def validate_frozen_sandbox(sandbox: DockerPythonSandbox) -> None:
    if sandbox.configuration != FROZEN_SANDBOX_CONFIGURATION:
        raise ValueError("FINAL functional sandbox differs from the exact frozen configuration")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _experiment_identity(expectations: FrozenExpectations) -> str:
    return _canonical_digest({
        "dataset": expectations.dataset_sha256,
        "split": expectations.split_sha256,
        "proposition": expectations.proposition_sha256,
        "evaluator": expectations.evaluator_sha256,
        "protocol": expectations.protocol_sha256,
        "final_manifest": expectations.final_manifest_sha256,
        "predictor": expectations.predictor_sha256,
        "policy": expectations.policy_sha256,
        "threshold": str(expectations.threshold),
        "candidates": expectations.candidates,
        "primary_baseline": expectations.primary_baseline,
    })


def _completed_duplicate(
    results_root: Path,
    experiment_identity: str,
    *,
    allowed_run_id: UUID | None = None,
) -> bool:
    if not results_root.exists():
        return False
    for status_path in results_root.glob("*/status.json"):
        if allowed_run_id is not None and status_path.parent.name == str(allowed_run_id):
            continue
        try:
            status = _read_object(status_path)
            manifest = _read_object(status_path.with_name("manifest.json"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        configuration = manifest.get("configuration", {})
        if (status.get("status") == "completed"
                and isinstance(configuration, dict)
                and configuration.get("final_experiment_identity") == experiment_identity):
            return True
    return False


def verify_final_freeze(
    *,
    paths: FinalPaths = FinalPaths(),
    expectations: FrozenExpectations = FrozenExpectations(),
    results_root: Path | None = None,
    explicit_authorization: bool = False,
    require_authorization: bool = True,
    allowed_run_id: UUID | None = None,
    protocol_loader=load_execution_protocol,
    provenance_verifier=None,
    repository_root: Path = REPOSITORY_ROOT,
) -> FreezeVerification:
    """Verify bytes and operational authorization without parsing FINAL tasks."""
    resolved_root = require_canonical_final_root(
        results_root, repository_root=repository_root)
    authorization_path = require_canonical_authorization_path(
        paths.authorization, repository_root=repository_root)
    try:
        tracked_authorization = _git(
            repository_root, "ls-files", "--error-unmatch", "--",
            str(CANONICAL_AUTHORIZATION_RELATIVE))
    except subprocess.CalledProcessError as exc:
        raise ValueError("canonical FINAL authorization artifact is not tracked by Git") from exc
    if tracked_authorization != str(CANONICAL_AUTHORIZATION_RELATIVE):
        raise ValueError("canonical FINAL authorization artifact is not tracked by Git")
    actual = {
        "dataset_sha256": sha256(paths.dataset),
        "split_manifest_sha256": sha256(paths.split),
        "proposition_specification_sha256": sha256(paths.proposition),
        "evaluator_manifest_sha256": sha256(paths.evaluator),
        "protocol_sha256": sha256(paths.protocol),
        "final_manifest_sha256": sha256(paths.final_manifest),
        "predictor_sha256": sha256(paths.predictor_directory / "predictor.pkl"),
        "policy_sha256": sha256(paths.policy),
    }
    expected = {
        "dataset_sha256": expectations.dataset_sha256,
        "split_manifest_sha256": expectations.split_sha256,
        "proposition_specification_sha256": expectations.proposition_sha256,
        "evaluator_manifest_sha256": expectations.evaluator_sha256,
        "protocol_sha256": expectations.protocol_sha256,
        "final_manifest_sha256": expectations.final_manifest_sha256,
        "predictor_sha256": expectations.predictor_sha256,
        "policy_sha256": expectations.policy_sha256,
    }
    mismatches = [name for name, value in actual.items() if value != expected[name]]
    if mismatches:
        raise ValueError("frozen identity mismatch: " + ", ".join(sorted(mismatches)))

    policy = FinalPolicy.model_validate_json(paths.policy.read_bytes())
    if (policy.policy_id != "routellm-frozen-final-policy"
            or policy.policy_version != POLICY_VERSION
            or policy.quality_threshold != expectations.threshold
            or policy.candidate_portfolio != expectations.candidates
            or policy.predictor.get("artifact")
            != "artifacts/routing-quality/rb12-train-candidate-v1"
            or policy.predictor.get("sha256") != expectations.predictor_sha256
            or policy.primary_fixed_baseline != expectations.primary_baseline
            or policy.selection.get("ordered_keys") != [
                "projected_cost_usd_ascending", "candidate_id_ascending"]
            or policy.fallback.get("ordered_keys") != [
                "predicted_acceptability_descending", "projected_cost_usd_ascending",
                "candidate_id_ascending"]
            or policy.serving_cost != {
                "counterfactual_matrix_cost_excluded": True,
                "evaluation_cost_excluded": True,
                "selected_candidate_realized_inference_cost_only": True,
            }):
        raise ValueError("frozen policy semantics mismatch")
    for implementation, digest in policy.implementation_files.items():
        implementation_path = Path(implementation)
        if not implementation_path.is_absolute():
            implementation_path = repository_root / implementation_path
        if sha256(implementation_path) != digest:
            raise ValueError(f"routing implementation hash mismatch: {implementation}")

    predictor_metadata = _read_object(paths.predictor_directory / "metadata.json")
    if (Decimal(str(predictor_metadata.get("approved_quality_threshold", THRESHOLD)))
            != expectations.threshold
            or tuple(predictor_metadata.get("known_candidate_ids", ()))
            != tuple(sorted(expectations.candidates))
            or predictor_metadata.get("model_sha256", predictor_metadata.get("predictor_sha256"))
            != expectations.predictor_sha256):
        raise ValueError("predictor approval metadata mismatch")

    protocol, models, protocol_digest = protocol_loader(paths.protocol)
    if protocol_digest != expectations.protocol_sha256:
        raise ValueError("Protocol 1.7 identity mismatch")
    if tuple(model.model_id for model in models) != expectations.candidates:
        raise ValueError("protocol candidate portfolio mismatch")
    if protocol.get("version") != "1.7.0":
        raise ValueError("FINAL requires Protocol 1.7")

    evaluator = _read_object(paths.evaluator)
    if evaluator.get("version") != "1.3.0":
        raise ValueError("FINAL requires Evaluator 1.3")
    for implementation, digest in evaluator.get("implementation_files", {}).items():
        implementation_path = Path(implementation)
        if not implementation_path.is_absolute():
            implementation_path = repository_root / implementation_path
        if sha256(implementation_path) != digest:
            raise ValueError(f"evaluator implementation hash mismatch: {implementation}")

    readiness = _read_object(paths.pricing_readiness)
    if (readiness.get("status") != READY_STATUS
            or readiness.get("protocol_sha256") != expectations.protocol_sha256
            or readiness.get("protocol_version") != "1.7.0"):
        raise ValueError("Protocol 1.7 pricing readiness is stale")
    readiness_ids = tuple(item.get("model_id") for item in readiness.get("models", []))
    if set(readiness_ids) != {*expectations.candidates, JUDGE_MODEL_ID}:
        raise ValueError("pricing readiness does not cover the frozen paid portfolio")

    authorization = _read_canonical_authorization(authorization_path)
    if authorization.get("version") != "1.0.0":
        raise ValueError("FINAL authorization schema version mismatch")
    if authorization.get("experiment") != {
        "dataset_sha256": expectations.dataset_sha256,
        "evaluator_manifest_sha256": expectations.evaluator_sha256,
        "final_manifest_sha256": expectations.final_manifest_sha256,
        "policy_sha256": expectations.policy_sha256,
        "predictor_sha256": expectations.predictor_sha256,
        "proposition_specification_sha256": expectations.proposition_sha256,
        "protocol_sha256": expectations.protocol_sha256,
        "split_manifest_sha256": expectations.split_sha256,
    }:
        raise ValueError("FINAL authorization identity mismatch")
    calls = authorization.get("call_limits", {})
    if calls != {"candidate_calls": EXPECTED_CANDIDATE_CALLS, "candidate_retries": 0,
                  "semantic_judge_calls": MAXIMUM_JUDGE_CALLS,
                  "semantic_judge_retries": 0}:
        raise ValueError("FINAL authorization call limits mismatch")
    if authorization.get("cost_budget_usd") != {
            "conservative_maximum": "1.55262531",
            "expected": "0.80157315",
            "repository_estimated_maximum": "1.41147755",
    }:
        raise ValueError("FINAL authorization cost budget mismatch")
    evidence = authorization.get("evidence", {})
    if (evidence.get("pricing_readiness_sha256") != sha256(paths.pricing_readiness)
            or evidence.get("gemini_native_minimal_confirmation_status_sha256")
            != sha256(paths.confirmation_status)):
        raise ValueError("FINAL authorization readiness evidence mismatch")
    confirmation = _read_object(paths.confirmation_status)
    if (confirmation.get("status") != "confirmed"
            or confirmation.get("calls_attempted") != 5
            or confirmation.get("complete_responses") != 5
            or confirmation.get("other_failures") != 0
            or confirmation.get("output_budget_exhaustions") != 0):
        raise ValueError("Gemini native minimal confirmation evidence is not successful")
    auth = authorization.get("authorization", {})
    authorization_status = str(auth.get("status", "MISSING"))
    authorized = bool(auth.get("final_paid_execution_authorized"))
    today = datetime.now(timezone.utc).date().isoformat()
    authorization_ready = bool(
        explicit_authorization and authorized
        and authorization_status == "AUTHORIZED_FOR_FINAL_EXECUTION"
        and auth.get("authorized_by")
        and auth.get("authorized_on") == today
        and auth.get("pricing_reverified_by")
        and auth.get("pricing_reverified_on") == today
    )
    if require_authorization and not authorization_ready:
        raise ValueError(
            "FINAL paid execution lacks same-day authorization and pricing reverification")
    git_commit = None
    source_hashes: dict[str, str] = {}
    source_set_sha256 = None
    if require_authorization and authorization_ready:
        verifier = provenance_verifier or verify_execution_provenance
        git_commit, source_hashes, source_set_sha256 = verifier(
            authorization, repository_root=repository_root)
    identity = _experiment_identity(expectations)
    duplicate = _completed_duplicate(
        resolved_root, identity, allowed_run_id=allowed_run_id)
    if duplicate:
        raise ValueError("completed FINAL result already exists for this frozen experiment")
    return FreezeVerification(
        status="READY" if authorization_ready else "BLOCKED",
        experiment_identity=identity, identities=actual,
        policy_sha256=actual["policy_sha256"],
        authorization_status=authorization_status,
        explicit_authorization=explicit_authorization,
        duplicate_completed_run=duplicate,
        git_commit=git_commit,
        source_hashes=source_hashes,
        source_set_sha256=source_set_sha256,
    )


def validate_final_run(
    root: Path,
    run_id: UUID,
    verification: FreezeVerification,
    paths: FinalPaths,
) -> BenchmarkRun:
    """Validate the complete persisted candidate matrix before paid judging/replay."""
    directory = root / str(run_id)
    status = _read_object(directory / "status.json")
    run = BenchmarkRun.model_validate_json((directory / "manifest.json").read_bytes())
    if status.get("status") != "completed" or run.run_id != run_id:
        raise ValueError("FINAL candidate run is not complete")
    if (run.configuration.get("final_experiment_identity")
            != verification.experiment_identity
            or run.configuration.get("final_policy_sha256") != verification.policy_sha256
            or run.configuration.get("execution_protocol_sha256")
            != verification.identities["protocol_sha256"]
            or run.configuration.get("execution_split") != "final"
            or run.configuration.get("final_authorization_status")
            != "AUTHORIZED_FOR_FINAL_EXECUTION"
            or run.configuration.get("source_dataset_sha256")
            != verification.identities["dataset_sha256"]
            or run.configuration.get("final_execution_git_commit")
            != verification.git_commit
            or run.configuration.get("final_source_hashes")
            != verification.source_hashes
            or run.configuration.get("final_source_set_sha256")
            != verification.source_set_sha256):
        raise ValueError("FINAL candidate run identity mismatch")
    feature_snapshots = run.configuration.get("request_features")
    feature_bindings = run.configuration.get("request_feature_bindings")
    if not isinstance(feature_snapshots, dict) or not isinstance(feature_bindings, dict):
        raise ValueError("FINAL candidate run lacks request-feature integrity bindings")
    tasks = {task.task_id: task for task in run.dataset.tasks}
    if (set(feature_snapshots) != set(run.selected_task_ids)
            or set(feature_bindings) != set(run.selected_task_ids)):
        raise ValueError("FINAL request-feature binding task set mismatch")
    for task_id in run.selected_task_ids:
        try:
            snapshot = RequestFeatures.model_validate(feature_snapshots[task_id])
            verify_request_feature_binding(tasks[task_id], snapshot, feature_bindings[task_id])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"FINAL request-feature binding failed for {task_id}") from exc
    if (verification.source_hashes
            and verification.source_hashes.get(
                "src/adaptive_llm_gateway/benchmarks/features.py")
            != feature_extractor_sha256()):
        raise ValueError("feature extractor differs from the authorized implementation")
    _, frozen_models, _ = load_execution_protocol(paths.protocol)
    if (len(run.selected_task_ids) != EXPECTED_REQUESTS
            or len(set(run.selected_task_ids)) != EXPECTED_REQUESTS
            or run.models != frozen_models):
        raise ValueError("FINAL candidate run shape mismatch")
    final_manifest = _read_object(paths.final_manifest)
    manifest_tasks = final_manifest.get("tasks")
    if not isinstance(manifest_tasks, list):
        raise ValueError("FINAL manifest task list is malformed")
    expected_task_ids = {item.get("task_id") for item in manifest_tasks
                         if isinstance(item, dict)}
    category_counts = Counter(task.category for task in run.dataset.tasks)
    if (set(run.selected_task_ids) != expected_task_ids
            or len(expected_task_ids) != EXPECTED_REQUESTS
            or category_counts != Counter({category: EXPECTED_CATEGORY_REQUESTS for category in (
                "qa", "summarization", "extraction", "classification", "json",
                "reasoning", "coding")})):
        raise ValueError("FINAL candidate run does not match the frozen assignment")
    result_paths = tuple(sorted((directory / "results").glob("*.json")))
    if len(result_paths) != EXPECTED_CANDIDATE_CALLS:
        raise ValueError("FINAL candidate matrix is incomplete")
    results = tuple(BenchmarkResult.model_validate_json(path.read_bytes())
                    for path in result_paths)
    pairs = {(result.task_id, result.model_id) for result in results}
    expected_pairs = {(task_id, model_id) for task_id in run.selected_task_ids
                      for model_id in CANDIDATES}
    if (any(result.run_id != run_id for result in results)
            or pairs != expected_pairs or len(pairs) != len(results)):
        raise ValueError("FINAL candidate matrix contains missing or duplicate rows")
    ledger = CandidateAttemptLedger(
        root / ".final-ledgers" / verification.experiment_identity
        / "candidate-attempts.json",
        experiment_identity=verification.experiment_identity,
    )
    ledger.initialize(run_id=run_id, task_bindings=feature_bindings,
                      candidate_ids=CANDIDATES)
    counts = ledger.counts()
    if counts["pending"] or counts["started"] or counts["completed"] + counts["failed"] != EXPECTED_CANDIDATE_CALLS:
        raise ValueError("FINAL candidate attempt ledger is incomplete or ambiguous")
    for result in results:
        durable = ledger.reusable(result.task_id, result.model_id, run_id)
        if durable is None or durable.model_dump(mode="json") != result.model_dump(mode="json"):
            raise ValueError("candidate result does not match its durable attempt ledger")
    return run


def _begin_semantic_judge_once(root: Path, run_id: UUID) -> None:
    directory = root / str(run_id)
    attempt_path = directory / "semantic-judge-attempt.json"
    marker = {
        "status": "started",
        "run_id": str(run_id),
        "judge_model": JUDGE_MODEL_ID,
        "evaluator_version": "1.3.0",
        "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
        "maximum_calls": MAXIMUM_JUDGE_CALLS,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if attempt_path.exists():
        existing = _read_object(attempt_path)
        if existing.get("run_id") != str(run_id):
            raise ValueError("semantic-judge marker belongs to another run")
        if existing.get("status") == "completed":
            return
    _atomic_write(attempt_path, (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode())


def _finish_semantic_judge(root: Path, run_id: UUID, summary: EvaluationSummary) -> None:
    marker = {
        "status": "completed",
        "run_id": str(run_id),
        "judge_model": JUDGE_MODEL_ID,
        "evaluator_version": "1.3.0",
        "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
        "maximum_calls": MAXIMUM_JUDGE_CALLS,
        "judge_calls": summary.judge_calls,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(
        root / str(run_id) / "semantic-judge-attempt.json",
        (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode(),
    )


def validate_final_evaluation(
    root: Path, run_id: UUID, verification: FreezeVerification | None = None,
) -> EvaluationSummary:
    """Reject evaluation artifacts produced by an older or incomplete workflow."""
    directory = root / str(run_id)
    marker = _read_object(directory / "semantic-judge-attempt.json")
    summary = EvaluationSummary.model_validate_json(
        (directory / "evaluation-summary.json").read_bytes())
    if (marker.get("status") != "completed" or summary.run_id != run_id
            or marker.get("judge_calls") != summary.judge_calls
            or summary.overall.evaluated_tasks != EXPECTED_CANDIDATE_CALLS
            or summary.judge_calls > MAXIMUM_JUDGE_CALLS):
        raise ValueError("FINAL evaluation is incomplete or exceeds the call budget")
    configuration = summary.evaluation_configuration
    functional = configuration.get("functional_sandbox")
    hybrid = configuration.get("hybrid_semantic_judge")
    if (functional != FROZEN_SANDBOX_CONFIGURATION
            or not isinstance(hybrid, dict)
            or hybrid != {
                "provider": "vercel",
                "model_id": JUDGE_UPSTREAM_MODEL,
                "judge_version": HYBRID_SEMANTIC_EVALUATOR_VERSION,
                "prompt_version": HYBRID_JUDGE_PROMPT_VERSION,
                "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
            }
            or configuration.get("semantic_judge") is not None):
        raise ValueError("FINAL evaluation configuration does not match Evaluator 1.3")
    evaluation_paths = tuple(sorted((directory / "evaluations").glob("*.json")))
    result_paths = tuple(sorted((directory / "results").glob("*.json")))
    if len(evaluation_paths) != EXPECTED_CANDIDATE_CALLS:
        raise ValueError("FINAL evaluation rows are incomplete")
    evaluations = tuple(EvaluationResult.model_validate_json(path.read_bytes())
                        for path in evaluation_paths)
    results = tuple(BenchmarkResult.model_validate_json(path.read_bytes())
                    for path in result_paths)
    result_by_id = {item.result_id: item for item in results}
    if (len(results) != EXPECTED_CANDIDATE_CALLS
            or len(result_by_id) != len(results)
            or {item.benchmark_result_id for item in evaluations} != set(result_by_id)):
        raise ValueError("FINAL evaluations do not match the canonical candidate results")
    for item in evaluations:
        result = result_by_id[item.benchmark_result_id]
        if (item.run_id != run_id or item.task_id != result.task_id
                or item.model_id != result.model_id):
            raise ValueError("FINAL evaluation identity differs from its candidate result")
    if sum(item.evaluation_call_made for item in evaluations) != summary.judge_calls:
        raise ValueError("FINAL semantic-judge call accounting is inconsistent")
    if verification is not None:
        judge_identity = _canonical_digest({
            "provider": "vercel", "model_id": JUDGE_UPSTREAM_MODEL,
            "judge_version": HYBRID_SEMANTIC_EVALUATOR_VERSION,
            "prompt_version": HYBRID_JUDGE_PROMPT_VERSION,
            "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
        })
        ledger = SemanticAttemptLedger(
            root / ".final-ledgers" / verification.experiment_identity
            / "semantic-attempts.json",
            experiment_identity=verification.experiment_identity,
            evaluator_identity=verification.identities["evaluator_manifest_sha256"],
            proposition_identity=verification.identities["proposition_specification_sha256"],
            judge_identity=judge_identity,
        )
        ledger.assert_resumable()
        counts = ledger.counts()
        if (counts["pending"] or counts["started"]
                or counts["completed"] + counts["failed"] != summary.judge_calls):
            raise ValueError("FINAL semantic-judge ledger is incomplete")
        for item in evaluations:
            if item.evaluation_call_made:
                durable = ledger.reusable(item.task_id, item.model_id)
                if (durable is None or durable.model_dump(mode="json")
                        != item.model_dump(mode="json")):
                    raise ValueError(
                        "FINAL semantic evaluation differs from its durable ledger")
    return summary


def _routing_features(values: dict[str, JsonValue]) -> RoutingRequestFeatures:
    data = dict(values)
    if data.get("category") == "json":
        data["category"] = "structured_json"
    return RoutingRequestFeatures.model_validate(data)


def _result_cost(result: BenchmarkResult) -> Decimal | None:
    if result.response is not None:
        return result.response.estimated_cost_usd
    value = result.error_details.get("estimated_cost_usd")
    try:
        cost = Decimal(str(value))
    except Exception:
        return None
    return cost if cost.is_finite() and cost >= 0 else None


def load_canonical_observations(
    root: Path,
    run: BenchmarkRun,
) -> tuple[CandidateObservation, ...]:
    """Rebuild replay rows from authoritative result and evaluation artifacts."""
    directory = root / str(run.run_id)
    result_paths = tuple(sorted((directory / "results").glob("*.json")))
    evaluation_paths = tuple(sorted((directory / "evaluations").glob("*.json")))
    results = tuple(BenchmarkResult.model_validate_json(path.read_bytes())
                    for path in result_paths)
    evaluations = tuple(EvaluationResult.model_validate_json(path.read_bytes())
                        for path in evaluation_paths)
    evaluation_by_result = {item.benchmark_result_id: item for item in evaluations}
    if (len(results) != EXPECTED_CANDIDATE_CALLS
            or len(evaluations) != EXPECTED_CANDIDATE_CALLS
            or len(evaluation_by_result) != len(evaluations)
            or set(evaluation_by_result) != {item.result_id for item in results}):
        raise ValueError("canonical FINAL result/evaluation artifact set is incomplete")
    tasks = {task.task_id: task for task in run.dataset.tasks}
    snapshots = run.configuration.get("request_features")
    bindings = run.configuration.get("request_feature_bindings")
    if not isinstance(snapshots, dict) or not isinstance(bindings, dict):
        raise ValueError("canonical FINAL feature snapshots are missing")
    observations = []
    for result in sorted(results, key=lambda item: (item.task_id, item.model_id)):
        evaluation = evaluation_by_result[result.result_id]
        if (evaluation.run_id != run.run_id or evaluation.task_id != result.task_id
                or evaluation.model_id != result.model_id):
            raise ValueError("canonical FINAL evaluation does not match its candidate result")
        task = tasks[result.task_id]
        snapshot = RequestFeatures.model_validate(snapshots[result.task_id])
        verify_request_feature_binding(task, snapshot, bindings[result.task_id])
        failure_type = evaluation.details.get("failure_type")
        observations.append(CandidateObservation(
            task_id=result.task_id,
            category=("structured_json" if snapshot.category == "json"
                      else str(snapshot.category)),
            model_id=result.model_id,
            request_features=snapshot.model_dump(mode="json"),
            provider_success=result.success,
            evaluation_status=evaluation.evaluation_status,
            acceptable=evaluation.acceptable,
            realized_cost_usd=_result_cost(result),
            judge_failure=failure_type in {"PROVIDER_FAILURE", "MALFORMED_JUDGMENT"},
        ))
    return tuple(observations)


def replay_router(
    observations: tuple[CandidateObservation, ...],
    *, predictor_path: Path,
    predictor_sha256: str,
    models: tuple[ModelConfig, ...],
    threshold: Decimal = THRESHOLD,
    feature_bindings: dict[str, dict[str, JsonValue]] | None = None,
) -> tuple[RouterSelection, ...]:
    """Pure offline replay. No provider object is constructed or called."""
    if threshold != THRESHOLD:
        raise ValueError("FINAL replay threshold must remain exactly 0.80")
    if sha256(predictor_path) != predictor_sha256:
        raise ValueError("frozen predictor checksum mismatch")
    if tuple(model.model_id for model in models) != CANDIDATES:
        raise ValueError("frozen replay candidate portfolio mismatch")
    pipeline = pickle.loads(predictor_path.read_bytes())
    if not isinstance(pipeline, Pipeline):
        raise ValueError("frozen predictor is not an sklearn pipeline")
    grouped: dict[str, list[CandidateObservation]] = defaultdict(list)
    for row in observations:
        grouped[row.task_id].append(row)
    if feature_bindings is not None and set(feature_bindings) != set(grouped):
        raise ValueError("replay request-feature binding task set mismatch")
    model_by_id = {model.model_id: model for model in models}
    policy = CostAwareRoutingPolicy()
    selections = []
    for task_id, rows in sorted(grouped.items()):
        if len(rows) != len(CANDIDATES) or {row.model_id for row in rows} != set(CANDIDATES):
            raise ValueError("every request requires exactly one row per frozen candidate")
        snapshots = {json.dumps(row.request_features, sort_keys=True) for row in rows}
        if len(snapshots) != 1 or len({row.category for row in rows}) != 1:
            raise ValueError("candidate rows disagree on request-visible features")
        if feature_bindings is not None:
            binding = feature_bindings.get(task_id)
            if (not isinstance(binding, dict)
                    or binding.get("task_id") != task_id
                    or binding.get("feature_extractor_sha256") != feature_extractor_sha256()
                    or binding.get("feature_snapshot_sha256")
                    != _canonical_digest(rows[0].request_features)):
                raise ValueError("replay request-feature binding mismatch")
        features = _routing_features(rows[0].request_features)
        canonical = tuple(canonical_from_production(features, model_by_id[model_id])
                          for model_id in CANDIDATES)
        probabilities = pipeline.predict_proba(canonical_feature_matrix(canonical))[:, 1]
        predictions = tuple(CandidatePrediction(
            model_id=model_id,
            predicted_acceptability=float(probability),
            projected_cost_usd=calculate_projected_cost(
                approximate_input_tokens=features.approximate_input_tokens,
                effective_max_output_tokens=resolve_effective_output_allowance(
                    features, model_by_id[model_id]),
                model=model_by_id[model_id],
            ),
        ) for model_id, probability in zip(CANDIDATES, probabilities, strict=True))
        decision = policy.route(predictions, threshold)
        selected = next(row for row in rows if row.model_id == decision.selected_model_id)
        selections.append(RouterSelection(
            task_id=task_id, category=selected.category,
            selected_model_id=selected.model_id,
            predicted_acceptability=decision.selected_predicted_acceptability,
            projected_cost_usd=decision.selected_projected_cost_usd,
            threshold_satisfied=decision.threshold_satisfied,
            fallback_used=decision.fallback_used,
            provider_success=selected.provider_success,
            evaluation_status=selected.evaluation_status,
            acceptable=selected.acceptable,
            realized_cost_usd=selected.realized_cost_usd,
            judge_failure=selected.judge_failure,
        ))
    return tuple(selections)


def _metrics(strategy_id: str, rows: list[CandidateObservation | RouterSelection], *,
             analysis_only: bool = False) -> StrategyMetrics:
    request_count = len(rows)
    fully = [row for row in rows if row.provider_success
             and row.evaluation_status == "evaluated" and row.acceptable is not None]
    acceptable = sum(row.acceptable is True for row in fully)
    failures = sum(not row.provider_success for row in rows)
    unresolved = request_count - len(fully) - failures
    costs = [row.realized_cost_usd for row in rows if row.realized_cost_usd is not None]
    total = sum(costs, Decimal(0))
    categories = tuple(_category_metrics(category, [row for row in rows
        if row.category == category]) for category in sorted({row.category for row in rows}))
    fallback_count = sum(bool(getattr(row, "fallback_used", False)) for row in rows)
    return StrategyMetrics(
        strategy_id=strategy_id, analysis_only=analysis_only,
        independent_requests=request_count, fully_evaluated=len(fully), acceptable=acceptable,
        task_acceptability=acceptable / request_count if request_count else 0,
        conditional_acceptability=acceptable / len(fully) if fully else None,
        provider_failures=failures,
        judge_failures=sum(row.judge_failure for row in rows),
        unresolved_evaluations=unresolved,
        total_realized_inference_cost_usd=total,
        mean_realized_cost_per_request_usd=(total / request_count if request_count else Decimal(0)),
        cost_complete=len(costs) == request_count,
        fallback_count=fallback_count,
        fallback_rate=(fallback_count / request_count if request_count else None),
        categories=categories,
    )


def _category_metrics(category: str, rows: list[CandidateObservation | RouterSelection]) -> CategoryMetrics:
    full = [row for row in rows if row.provider_success
            and row.evaluation_status == "evaluated" and row.acceptable is not None]
    acceptable = sum(row.acceptable is True for row in full)
    failures = sum(not row.provider_success for row in rows)
    costs = [row.realized_cost_usd for row in rows if row.realized_cost_usd is not None]
    total = sum(costs, Decimal(0))
    return CategoryMetrics(
        category=category, independent_requests=len(rows), fully_evaluated=len(full),
        acceptable=acceptable, task_acceptability=acceptable / len(rows) if rows else 0,
        conditional_acceptability=acceptable / len(full) if full else None,
        provider_failures=failures, judge_failures=sum(row.judge_failure for row in rows),
        unresolved_evaluations=len(rows) - len(full) - failures,
        total_realized_inference_cost_usd=total,
        mean_realized_cost_per_request_usd=total / len(rows) if rows else Decimal(0),
        cost_complete=len(costs) == len(rows),
    )


def aggregate_strategies(
    observations: tuple[CandidateObservation, ...],
    selections: tuple[RouterSelection, ...],
) -> tuple[StrategyMetrics, ...]:
    task_ids = {row.task_id for row in observations}
    if len(selections) != len(task_ids) or {row.task_id for row in selections} != task_ids:
        raise ValueError("router must select exactly one row per independent request")
    output = [_metrics("routellm", list(selections))]
    for model_id in CANDIDATES:
        output.append(_metrics(f"fixed:{model_id}", [row for row in observations
                                                     if row.model_id == model_id]))
    oracle_rows = []
    grouped: dict[str, list[CandidateObservation]] = defaultdict(list)
    for row in observations:
        grouped[row.task_id].append(row)
    for rows in grouped.values():
        acceptable = [row for row in rows if row.provider_success
                      and row.evaluation_status == "evaluated" and row.acceptable is True
                      and row.realized_cost_usd is not None]
        if acceptable:
            oracle_rows.append(min(acceptable,
                key=lambda row: (row.realized_cost_usd, row.model_id)))
    output.append(_metrics("oracle:cheapest-acceptable", oracle_rows, analysis_only=True))
    return tuple(output)


def build_final_results(
    *, verification: FreezeVerification,
    observations: tuple[CandidateObservation, ...],
    selections: tuple[RouterSelection, ...],
    judge_call_count: int,
    judge_evaluation_cost_usd: Decimal = Decimal(0),
    git_commit: str,
    generated_at: datetime | None = None,
) -> FinalResults:
    if not 0 <= judge_call_count <= MAXIMUM_JUDGE_CALLS:
        raise ValueError("semantic judge call count exceeds the frozen maximum")
    strategies = aggregate_strategies(observations, selections)
    router = next(item for item in strategies if item.strategy_id == "routellm")
    baseline = next(item for item in strategies
                    if item.strategy_id == f"fixed:{PRIMARY_BASELINE}")
    n = len({row.task_id for row in observations})
    categories = dict(sorted(Counter(row.category for row in selections).items()))
    cost_complete = router.cost_complete and baseline.cost_complete
    reduction = None
    if cost_complete and baseline.total_realized_inference_cost_usd > 0:
        with localcontext() as context:
            context.prec = 50
            reduction = float(Decimal(100) * (
                baseline.total_realized_inference_cost_usd
                - router.total_realized_inference_cost_usd
            ) / baseline.total_realized_inference_cost_usd)
    evaluation_complete = router.fully_evaluated == n
    y = router.task_acceptability * 100
    statement = None
    if (reduction is not None and n == EXPECTED_REQUESTS
            and len(categories) == 7
            and all(value == EXPECTED_CATEGORY_REQUESTS for value in categories.values())
            and verification.status == "READY"):
        statement = (
            f"Reduced LLM inference cost by {reduction:.2f}% versus fixed Claude Sonnet 5 "
            f"while maintaining {y:.2f}% task acceptability across {n} held-out requests "
            "in an offline held-out routing evaluation."
        )
    primary = PrimaryMetrics(
        cost_reduction_percent=reduction, task_acceptability_percent=y,
        independent_requests=n, cost_complete=cost_complete,
        evaluation_complete=evaluation_complete, resume_statement=statement,
    )
    matrix_costs = [row.realized_cost_usd for row in observations
                    if row.realized_cost_usd is not None]
    cost_accounting = CostAccounting(
        serving_cost_definition="realized inference cost of the selected candidate row",
        candidate_matrix_collection_cost_usd=sum(matrix_costs, Decimal(0)),
        candidate_matrix_cost_complete=len(matrix_costs) == len(observations),
        semantic_judge_cost_usd=judge_evaluation_cost_usd,
        excluded_from_primary_cost_comparison=(
            "counterfactual candidate generation",
            "semantic judging",
            "functional evaluation and other research overhead",
        ),
    )
    limitations = (
        "Offline held-out routing evaluation; this is not production traffic.",
        "Candidate-matrix collection cost and semantic-judge cost are research overhead and are excluded from serving-cost comparisons.",
        "Provider failures and unresolved selected evaluations count as non-acceptable in the primary task-acceptability denominator.",
        "The retrospective oracle is analysis-only and is not a deployable or headline baseline.",
    )
    return FinalResults(
        experiment_identity=verification.experiment_identity,
        identities=verification.identities, git_commit=git_commit,
        source_hashes=verification.source_hashes,
        source_set_sha256=verification.source_set_sha256,
        generated_at=generated_at or datetime.now(timezone.utc), threshold=THRESHOLD,
        candidate_portfolio=CANDIDATES, primary_baseline=PRIMARY_BASELINE,
        final_request_count=n, category_counts=categories,
        candidate_call_count=len(observations), judge_call_count=judge_call_count,
        strategies=strategies, router_selections=selections,
        primary_metrics=primary, cost_accounting=cost_accounting,
        evaluation_complete=evaluation_complete,
        cost_complete=cost_complete, limitations=limitations,
    )


def render_markdown(results: FinalResults) -> str:
    by_id = {item.strategy_id: item for item in results.strategies}
    baseline = by_id[f"fixed:{PRIMARY_BASELINE}"]
    labels = [
        ("RoutLLM", "routellm"),
        ("Fixed Nemotron", f"fixed:{CANDIDATES[0]}"),
        ("Fixed GPT-6 Luna", f"fixed:{CANDIDATES[1]}"),
        ("Fixed Gemini 3 Flash", f"fixed:{CANDIDATES[2]}"),
        ("Fixed Claude Sonnet 5", f"fixed:{CANDIDATES[3]}"),
        ("Oracle (analysis only)", "oracle:cheapest-acceptable"),
    ]
    lines = [
        "# RoutLLM FINAL Offline Held-Out Evaluation", "",
        "> This is an offline held-out routing evaluation, not production traffic.", "",
        "| Strategy | Task acceptability | Total inference cost | Mean cost/request | Cost reduction vs fixed Sonnet | Fallback rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, strategy_id in labels:
        metric = by_id[strategy_id]
        reduction = None
        if (metric.cost_complete and baseline.cost_complete
                and baseline.total_realized_inference_cost_usd > 0):
            reduction = 100 * float((baseline.total_realized_inference_cost_usd
                - metric.total_realized_inference_cost_usd)
                / baseline.total_realized_inference_cost_usd)
        lines.append(
            f"| {label} | {metric.task_acceptability * 100:.2f}% "
            f"({metric.acceptable}/{metric.independent_requests}) | "
            f"${metric.total_realized_inference_cost_usd:.8f} | "
            f"${metric.mean_realized_cost_per_request_usd:.8f} | "
            f"{f'{reduction:.2f}%' if reduction is not None else 'suppressed: incomplete cost'} | "
            f"{f'{metric.fallback_rate * 100:.2f}%' if metric.fallback_rate is not None else 'n/a'} |"
        )
    lines.extend(["", "## Category-level RoutLLM results", "",
        "| Category | N | Fully evaluated | Acceptable | Task acceptability | Cost complete |",
        "| --- | ---: | ---: | ---: | ---: | --- |"])
    for item in by_id["routellm"].categories:
        lines.append(f"| {item.category} | {item.independent_requests} | "
                     f"{item.fully_evaluated} | {item.acceptable} | "
                     f"{item.task_acceptability * 100:.2f}% | "
                     f"{'yes' if item.cost_complete else 'no'} |")
    lines.extend(["", "## Resume statement", ""])
    lines.append(results.primary_metrics.resume_statement or
        "Suppressed: cost coverage, request count, category count, or identity verification is incomplete.")
    lines.extend(["", "## Research overhead (excluded from X)", "",
        f"- Candidate-matrix collection: ${results.cost_accounting.candidate_matrix_collection_cost_usd:.8f}"
        f" ({'complete' if results.cost_accounting.candidate_matrix_cost_complete else 'incomplete'} coverage)",
        f"- Semantic judging: ${results.cost_accounting.semantic_judge_cost_usd:.8f}"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in results.limitations)
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_results(results: FinalResults, output: Path) -> tuple[Path, Path]:
    json_path = output / "final-results.json"
    report_path = output / "final-report.md"
    canonical_bytes = (json.dumps(results.model_dump(mode="json"), indent=2,
                                  sort_keys=True) + "\n").encode()
    if report_path.exists() and not json_path.exists():
        raise ValueError("FINAL report exists without its canonical JSON source")
    canonical = results
    if json_path.exists():
        canonical = FinalResults.model_validate_json(json_path.read_bytes())
        existing_payload = canonical.model_dump(mode="json")
        proposed_payload = results.model_dump(mode="json")
        existing_payload.pop("generated_at", None)
        proposed_payload.pop("generated_at", None)
        if existing_payload != proposed_payload:
            raise ValueError("existing canonical FINAL result differs and will not be overwritten")
    else:
        _atomic_write(json_path, canonical_bytes)
    expected_report = render_markdown(canonical).encode()
    if report_path.exists():
        if report_path.read_bytes() != expected_report:
            raise ValueError("existing FINAL report does not match canonical JSON")
    else:
        _atomic_write(report_path, expected_report)
    return json_path, report_path


def recover_results_publication(output: Path) -> tuple[Path, Path]:
    """Regenerate only the presentation layer from already-canonical JSON."""
    json_path = output / "final-results.json"
    report_path = output / "final-report.md"
    if not json_path.is_file():
        raise ValueError("canonical FINAL JSON does not exist for publication recovery")
    canonical = FinalResults.model_validate_json(json_path.read_bytes())
    expected_report = render_markdown(canonical).encode()
    if report_path.exists() and report_path.read_bytes() != expected_report:
        raise ValueError("existing FINAL report does not match canonical JSON")
    if not report_path.exists():
        _atomic_write(report_path, expected_report)
    return json_path, report_path


async def _semantic_plan(root: Path, run_id: UUID) -> tuple[tuple[str, str], ...]:
    service = EvaluationService(root)
    run, results = await service.repository.load_run(run_id)
    task_by_id = {task.task_id: task for task in run.dataset.tasks}
    pairs = []
    for result in results:
        task = task_by_id[result.task_id]
        static = evaluator_for(task).evaluate(task, result)
        if (static.evaluation_status == "requires_semantic_judge"
                and result.response is not None):
            pairs.append((task.task_id, result.model_id))
    return tuple(pairs)


async def _semantic_evaluate(
    service: EvaluationService,
    ledger: SemanticAttemptLedger,
    run_id: UUID,
) -> EvaluationSummary:
    """Run frozen evaluators with FINAL-only durable orchestration around them."""
    run, results = await service.repository.load_run(run_id)
    task_by_id = {task.task_id: task for task in run.dataset.tasks}
    ledger.assert_resumable()
    evaluation_directory = service.repository._directory(run_id) / "evaluations"
    existing: dict[UUID, EvaluationResult] = {}
    for path in sorted(evaluation_directory.glob("*.json")):
        item = EvaluationResult.model_validate_json(path.read_bytes())
        if item.run_id != run_id or item.benchmark_result_id in existing:
            raise EvaluationArtifactError("Persisted evaluation identity is invalid")
        existing[item.benchmark_result_id] = item
    evaluations = []
    for result in results:
        task = task_by_id[result.task_id]
        persisted = existing.get(result.result_id)
        if persisted is not None:
            if persisted.task_id != result.task_id or persisted.model_id != result.model_id:
                raise EvaluationArtifactError("Persisted evaluation does not match its result")
            if persisted.evaluation_call_made:
                durable = ledger.reusable(task.task_id, result.model_id)
                if (durable is None or durable.model_dump(mode="json")
                        != persisted.model_dump(mode="json")):
                    raise EvaluationArtifactError(
                        "Persisted semantic evaluation is not bound to its ledger")
            evaluations.append(persisted)
            continue
        try:
            evaluation = evaluator_for(task).evaluate(task, result)
            if (evaluation.evaluation_status == "requires_functional_execution"
                    and service.functional_sandbox is not None
                    and result.response is not None):
                evaluation = await service._evaluate_functionally(task, result, evaluation)
            if (evaluation.evaluation_status == "requires_semantic_judge"
                    and result.response is not None):
                durable = ledger.reusable(task.task_id, result.model_id)
                if durable is not None:
                    if durable.benchmark_result_id != result.result_id:
                        raise EvaluationArtifactError(
                            "Semantic ledger result does not match the benchmark result")
                    evaluation = durable
                else:
                    ledger.start(task.task_id, result.model_id)
                    evaluation = await service._evaluate_hybrid_semantically(
                        task, result, evaluation)
            _atomic_write(
                evaluation_directory / f"{evaluation.benchmark_result_id}.json",
                (evaluation.model_dump_json(indent=2) + "\n").encode())
            if evaluation.evaluation_call_made:
                ledger.finish(evaluation)
            evaluations.append(evaluation)
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationArtifactError(
                f"Task {task.task_id!r} has invalid evaluation metadata") from exc
    try:
        summary = aggregate(
            run, results, evaluations, evaluation_configuration=service.configuration)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationArtifactError(
            "Benchmark evaluations could not be aggregated") from exc
    _atomic_write(
        service.repository._directory(run_id) / "evaluation-summary.json",
        (summary.model_dump_json(indent=2) + "\n").encode())
    return summary


async def _prepare_semantic_evaluation(
    root: Path,
    run_id: UUID,
    paths: FinalPaths,
    verification: FreezeVerification,
    sandbox_image: str,
) -> tuple[EvaluationService, SemanticAttemptLedger, tuple[tuple[str, str], ...]]:
    """Validate every local prerequisite before any paid attempt is claimed."""
    sandbox = DockerPythonSandbox(image=sandbox_image)
    validate_frozen_sandbox(sandbox)
    settings = GatewaySettings.from_environment()
    if settings.api_key is None:
        raise ValueError("AI_GATEWAY_API_KEY is required for paid semantic judging")
    specification = PropositionSpecification.model_validate_json(paths.proposition.read_bytes())
    plan = await _semantic_plan(root, run_id)
    if len(plan) > MAXIMUM_JUDGE_CALLS:
        raise ValueError("semantic-judge plan exceeds the frozen maximum")
    judge_identity = _canonical_digest({
        "provider": "vercel", "model_id": JUDGE_UPSTREAM_MODEL,
        "judge_version": HYBRID_SEMANTIC_EVALUATOR_VERSION,
        "prompt_version": HYBRID_JUDGE_PROMPT_VERSION,
        "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
    })
    ledger = SemanticAttemptLedger(
        root / ".final-ledgers" / verification.experiment_identity
        / "semantic-attempts.json",
        experiment_identity=verification.experiment_identity,
        evaluator_identity=verification.identities["evaluator_manifest_sha256"],
        proposition_identity=verification.identities["proposition_specification_sha256"],
        judge_identity=judge_identity,
    )
    ledger.initialize(run_id=run_id, pairs=plan)
    model = next(model for model in JUDGE_SELECTION_MODELS if model.model_id == JUDGE_MODEL_ID)
    judge = VercelHybridSemanticJudge(
        VercelGatewayProvider(model, settings), provider_name="vercel",
        model_id=JUDGE_UPSTREAM_MODEL, max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS)
    service = EvaluationService(
        root, functional_sandbox=sandbox, hybrid_semantic_judge=judge,
        proposition_specification=specification)
    return service, ledger, plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--authorize-final", action="store_true")
    preflight.add_argument("--require-authorization", action="store_true")
    preflight.add_argument("--results-root", type=Path)

    semantic = sub.add_parser("semantic-judge")
    semantic.add_argument("--root", type=Path)
    semantic.add_argument("--run-id", type=UUID, required=True)
    semantic.add_argument("--dry-run", action="store_true")
    semantic.add_argument("--execute", action="store_true")
    semantic.add_argument("--allow-paid-judge", action="store_true")
    semantic.add_argument("--authorize-final", action="store_true")
    semantic.add_argument("--sandbox-image", default=DEFAULT_SANDBOX_IMAGE)

    replay = sub.add_parser("replay")
    replay.add_argument("--root", type=Path)
    replay.add_argument("--run-id", type=UUID, required=True)
    replay.add_argument("--output", type=Path)
    replay.add_argument("--authorize-final", action="store_true")
    args = parser.parse_args()
    paths = FinalPaths()
    supplied_root = getattr(args, "root", None)
    try:
        root = require_canonical_final_root(supplied_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.command == "preflight":
        report = verify_final_freeze(
            paths=paths, results_root=args.results_root,
            explicit_authorization=args.authorize_final,
            require_authorization=args.require_authorization)
        print(report.model_dump_json(indent=2))
        return
    if args.command == "semantic-judge":
        if args.dry_run:
            print(json.dumps({"status": "DRY_RUN", "judge_model": JUDGE_MODEL_ID,
                "evaluator_version": "1.3.0", "max_output_tokens": 768,
                "maximum_calls": MAXIMUM_JUDGE_CALLS, "provider_calls_made": 0},
                indent=2, sort_keys=True))
            return
        if not (args.execute and args.allow_paid_judge and args.authorize_final):
            parser.error("paid semantic judging requires --execute, --allow-paid-judge, and --authorize-final")
        verification = verify_final_freeze(
            paths=paths, results_root=root, explicit_authorization=True,
            require_authorization=True, allowed_run_id=args.run_id)
        validate_final_run(root, args.run_id, verification, paths)
        service, ledger, _ = asyncio.run(_prepare_semantic_evaluation(
            root, args.run_id, paths, verification, args.sandbox_image))
        _begin_semantic_judge_once(root, args.run_id)
        summary = asyncio.run(_semantic_evaluate(service, ledger, args.run_id))
        _finish_semantic_judge(root, args.run_id, summary)
        return

    output = canonical_publication_directory(REPOSITORY_ROOT, args.run_id)
    if args.output is not None and _literal_absolute(args.output) != output:
        parser.error("FINAL publication output must use the canonical run directory")
    if (output / "final-results.json").is_file():
        paths_written = recover_results_publication(output)
        print(json.dumps({"final_results": str(paths_written[0]),
                          "report": str(paths_written[1]),
                          "publication_recovered": True}, sort_keys=True))
        return

    verification = verify_final_freeze(
        paths=paths, results_root=root, explicit_authorization=args.authorize_final,
        require_authorization=True, allowed_run_id=args.run_id)
    run = validate_final_run(root, args.run_id, verification, paths)
    summary = validate_final_evaluation(root, args.run_id, verification)
    observations = load_canonical_observations(root, run)
    _, models, _ = load_execution_protocol(paths.protocol)
    selections = replay_router(
        observations,
        predictor_path=paths.predictor_directory / "predictor.pkl",
        predictor_sha256=FrozenExpectations().predictor_sha256,
        models=models,
        feature_bindings=run.configuration["request_feature_bindings"],
    )
    results = build_final_results(
        verification=verification, observations=observations, selections=selections,
        judge_call_count=summary.judge_calls,
        judge_evaluation_cost_usd=summary.judge_evaluation_cost_usd,
        git_commit=verification.git_commit or "")
    paths_written = write_results(results, output)
    print(json.dumps({"final_results": str(paths_written[0]),
                      "report": str(paths_written[1])}, sort_keys=True))


if __name__ == "__main__":
    main()
