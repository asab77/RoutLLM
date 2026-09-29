"""Versioned trusted-artifact loading and production quality prediction."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import sklearn
from pydantic import ConfigDict, ValidationError
from sklearn.pipeline import Pipeline

from adaptive_llm_gateway.errors import (
    CorruptPredictorArtifactError,
    IncompatibleArtifactFormatError,
    IncompatibleCategoryTaxonomyError,
    IncompatibleFeatureSchemaError,
    IncompatiblePredictorFormulationError,
    MissingRoutingCategoryError,
    PredictorArtifactChecksumError,
    PredictorArtifactNotFoundError,
    PredictorInputCompatibilityError,
    UnsupportedPredictorCandidateError,
)
from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.models.schemas import DomainModel, Identifier

from .features import (
    ROUTING_CATEGORY_TAXONOMY_VERSION,
    RoutingCategory,
    RoutingRequestFeatures,
)
from .policy import ModelAcceptabilityPrediction
from .quality_features import (
    CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION,
    PREDICTOR_FORMULATION_ID,
    PREDICTOR_FORMULATION_VERSION,
    QUALITY_PREPROCESSING_ID,
    canonical_feature_matrix,
    canonical_from_production,
)

ARTIFACT_FORMAT_VERSION = "1.0.0"
ARTIFACT_METADATA_FILENAME = "metadata.json"
ARTIFACT_MODEL_FILENAME = "predictor.pkl"
PRODUCTION_ARTIFACT_FORMAT_VERSION = "routellm-production-router-v1"
PRODUCTION_PREDICTOR_SHA256 = "502db83a54c4072c9741a8e4c406498ad88ec97e03bce1ddaf3e0a1b0001c0aa"
PRODUCTION_PROTOCOL_VERSION = "1.7.0"
PRODUCTION_PROTOCOL_SHA256 = "b4cde3954a8ccd1b54684dcb62da303e7bc806248306464ae536b9ff717a0bd8"
PRODUCTION_SPLIT_SHA256 = "98c639be29da4e11fdf48073d74e83805f16fdfb72b0102b5f5543213ab4960b"
PRODUCTION_TRAINING_DATASET_SHA256 = "1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005"
PRODUCTION_SOURCE_RUN_ID = "cc28470f-55ed-4a2f-a6ee-8d25cf93ecd2"
PRODUCTION_QUALITY_THRESHOLD = 0.80
FOUNDATION_V3_SHA256 = "64eda8e7388233f645906412a2524982ebfb9c848c42ee24a468ba04c31d16e6"
FOUNDATION_V3_PROTOCOL_SHA256 = "973c46e2dfd7c5739f623ff1c7f2378dc3e77c80e5815e23c975816700d4b861"
PHASE_8C0_FORMULATION_EVIDENCE = "PHASE_8C_0_PROVIDER_PIN_ABLATION"
FOUNDATION_V3_CANDIDATE_IDS = (
    "candidate-claude-sonnet-5",
    "candidate-gemini-3-flash",
    "candidate-gpt-6-luna",
    "candidate-nemotron-3.5-lightning",
)


class QualityPredictorArtifactMetadata(DomainModel):
    artifact_format_version: str
    predictor_formulation_id: str
    predictor_formulation_version: str
    canonical_feature_schema_version: str
    category_taxonomy_version: str
    training_dataset_name: str
    training_dataset_version: str
    training_dataset_sha256: str
    foundation_v3_protocol_sha256: str
    formulation_evidence: str
    valid_training_rows: int
    missing_label_rows: int
    known_candidate_ids: tuple[Identifier, ...]
    known_categories: tuple[RoutingCategory, ...]
    sklearn_version: str
    python_version: str
    preprocessing_id: str
    build_entrypoint: str
    model_filename: str
    model_sha256: str


class ProductionRuntimeCompatibility(DomainModel):
    python_major_minor: str
    scikit_learn_version: str


class ProductionDeploymentApproval(DomainModel):
    decision: str
    phase: str
    statement: str


class SourceValidationMetadata(DomainModel):
    artifact_format_version: str
    deployment_status: str
    missing_label_rows: int
    source_run_id: str
    split_sha256: str
    training_dataset_sha256: str
    valid_training_rows: int


class ProductionQualityPredictorArtifactMetadata(DomainModel):
    """Narrow deployment approval wrapped around the unchanged Phase 9 bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    approved_quality_threshold: float
    artifact_format_version: str
    canonical_feature_schema_version: str
    deployment_approval: ProductionDeploymentApproval
    known_candidate_ids: tuple[Identifier, ...]
    known_categories: tuple[RoutingCategory, ...]
    model_filename: str
    predictor_formulation_id: str
    predictor_formulation_version: str
    predictor_sha256: str
    preprocessing_id: str
    protocol_sha256: str
    protocol_version: str
    runtime_compatibility: ProductionRuntimeCompatibility
    source_validation_metadata: SourceValidationMetadata


def expected_metadata(
    *,
    model_sha256: str,
    known_candidate_ids: tuple[str, ...],
    known_categories: tuple[RoutingCategory, ...],
    valid_training_rows: int,
    missing_label_rows: int,
) -> QualityPredictorArtifactMetadata:
    return QualityPredictorArtifactMetadata(
        artifact_format_version=ARTIFACT_FORMAT_VERSION,
        predictor_formulation_id=PREDICTOR_FORMULATION_ID,
        predictor_formulation_version=PREDICTOR_FORMULATION_VERSION,
        canonical_feature_schema_version=CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION,
        category_taxonomy_version=ROUTING_CATEGORY_TAXONOMY_VERSION,
        training_dataset_name="routellm-foundation-v3",
        training_dataset_version="3.0.0",
        training_dataset_sha256=FOUNDATION_V3_SHA256,
        foundation_v3_protocol_sha256=FOUNDATION_V3_PROTOCOL_SHA256,
        formulation_evidence=PHASE_8C0_FORMULATION_EVIDENCE,
        valid_training_rows=valid_training_rows,
        missing_label_rows=missing_label_rows,
        known_candidate_ids=known_candidate_ids,
        known_categories=known_categories,
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
        preprocessing_id=QUALITY_PREPROCESSING_ID,
        build_entrypoint="python -m adaptive_llm_gateway.routing.train_predictor",
        model_filename=ARTIFACT_MODEL_FILENAME,
        model_sha256=model_sha256,
    )


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


def write_trusted_quality_artifact(
    output_directory: Path,
    pipeline: Pipeline,
    *,
    known_candidate_ids: tuple[str, ...],
    known_categories: tuple[RoutingCategory, ...],
    valid_training_rows: int,
    missing_label_rows: int,
) -> tuple[Path, Path, QualityPredictorArtifactMetadata]:
    """Serialize application-owned sklearn output and its validated metadata."""
    model_bytes = pickle.dumps(pipeline, protocol=5)
    checksum = hashlib.sha256(model_bytes).hexdigest()
    metadata = expected_metadata(
        model_sha256=checksum,
        known_candidate_ids=known_candidate_ids,
        known_categories=known_categories,
        valid_training_rows=valid_training_rows,
        missing_label_rows=missing_label_rows,
    )
    model_path = output_directory / ARTIFACT_MODEL_FILENAME
    metadata_path = output_directory / ARTIFACT_METADATA_FILENAME
    _atomic_write(model_path, model_bytes)
    _atomic_write(
        metadata_path,
        (json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(),
    )
    return model_path, metadata_path, metadata


def _validate_metadata(metadata: QualityPredictorArtifactMetadata) -> None:
    if metadata.artifact_format_version != ARTIFACT_FORMAT_VERSION:
        raise IncompatibleArtifactFormatError("unsupported predictor artifact format")
    if (
        metadata.predictor_formulation_id != PREDICTOR_FORMULATION_ID
        or metadata.predictor_formulation_version != PREDICTOR_FORMULATION_VERSION
    ):
        raise IncompatiblePredictorFormulationError("predictor formulation mismatch")
    if metadata.canonical_feature_schema_version != CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION:
        raise IncompatibleFeatureSchemaError("canonical feature schema mismatch")
    if metadata.category_taxonomy_version != ROUTING_CATEGORY_TAXONOMY_VERSION:
        raise IncompatibleCategoryTaxonomyError("category taxonomy mismatch")
    if (
        metadata.training_dataset_sha256 != FOUNDATION_V3_SHA256
        or metadata.foundation_v3_protocol_sha256 != FOUNDATION_V3_PROTOCOL_SHA256
        or metadata.formulation_evidence != PHASE_8C0_FORMULATION_EVIDENCE
        or metadata.preprocessing_id != QUALITY_PREPROCESSING_ID
    ):
        raise CorruptPredictorArtifactError("artifact governance metadata mismatch")
    if tuple(metadata.known_candidate_ids) != FOUNDATION_V3_CANDIDATE_IDS:
        raise CorruptPredictorArtifactError("artifact candidate compatibility set mismatch")
    if set(metadata.known_categories) != set(RoutingCategory):
        raise CorruptPredictorArtifactError("artifact category compatibility set mismatch")
    if (
        metadata.training_dataset_name != "routellm-foundation-v3"
        or metadata.training_dataset_version != "3.0.0"
        or metadata.valid_training_rows != 216
        or metadata.missing_label_rows != 8
        or metadata.model_filename != ARTIFACT_MODEL_FILENAME
        or len(metadata.model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in metadata.model_sha256)
    ):
        raise CorruptPredictorArtifactError("artifact build metadata mismatch")
    if metadata.sklearn_version != sklearn.__version__:
        raise IncompatibleArtifactFormatError("artifact sklearn runtime mismatch")
    if metadata.python_version.split(".")[:2] != platform.python_version().split(".")[:2]:
        raise IncompatibleArtifactFormatError("artifact Python runtime mismatch")


def _validate_production_metadata(
    metadata: ProductionQualityPredictorArtifactMetadata,
) -> None:
    source = metadata.source_validation_metadata
    approval = metadata.deployment_approval
    runtime = metadata.runtime_compatibility
    if metadata.artifact_format_version != PRODUCTION_ARTIFACT_FORMAT_VERSION:
        raise IncompatibleArtifactFormatError("unsupported production artifact format")
    if metadata.predictor_sha256 != PRODUCTION_PREDICTOR_SHA256:
        raise PredictorArtifactChecksumError("unapproved production predictor checksum")
    if metadata.canonical_feature_schema_version != CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION:
        raise IncompatibleFeatureSchemaError("canonical feature schema mismatch")
    if (
        metadata.predictor_formulation_id != PREDICTOR_FORMULATION_ID
        or metadata.predictor_formulation_version != PREDICTOR_FORMULATION_VERSION
        or metadata.preprocessing_id != QUALITY_PREPROCESSING_ID
    ):
        raise IncompatiblePredictorFormulationError("predictor formulation mismatch")
    if tuple(metadata.known_candidate_ids) != FOUNDATION_V3_CANDIDATE_IDS:
        raise CorruptPredictorArtifactError("artifact candidate compatibility set mismatch")
    if set(metadata.known_categories) != set(RoutingCategory):
        raise IncompatibleCategoryTaxonomyError("artifact category compatibility set mismatch")
    if (
        metadata.approved_quality_threshold != PRODUCTION_QUALITY_THRESHOLD
        or metadata.model_filename != ARTIFACT_MODEL_FILENAME
        or metadata.protocol_version != PRODUCTION_PROTOCOL_VERSION
        or metadata.protocol_sha256 != PRODUCTION_PROTOCOL_SHA256
    ):
        raise CorruptPredictorArtifactError("production approval metadata mismatch")
    if (
        approval.decision != "APPROVED_FROZEN_PRODUCTION_ROUTER"
        or approval.phase != "12.1B"
        or not approval.statement
    ):
        raise CorruptPredictorArtifactError("production deployment approval is missing")
    if (
        source.artifact_format_version != "phase9-dev-candidate-v1"
        or source.deployment_status != "DEV_VALIDATION_CANDIDATE_NOT_DEPLOYED"
        or source.split_sha256 != PRODUCTION_SPLIT_SHA256
        or source.training_dataset_sha256 != PRODUCTION_TRAINING_DATASET_SHA256
        or source.source_run_id != PRODUCTION_SOURCE_RUN_ID
        or source.valid_training_rows != 557
        or source.missing_label_rows != 3
    ):
        raise CorruptPredictorArtifactError("source validation metadata mismatch")
    if runtime.scikit_learn_version != sklearn.__version__:
        raise IncompatibleArtifactFormatError("artifact sklearn runtime mismatch")
    if runtime.python_major_minor != ".".join(platform.python_version_tuple()[:2]):
        raise IncompatibleArtifactFormatError("artifact Python runtime mismatch")


def load_trusted_quality_artifact(
    artifact_directory: Path,
) -> tuple[
    Pipeline,
    QualityPredictorArtifactMetadata | ProductionQualityPredictorArtifactMetadata,
]:
    """Load trusted local build output after metadata and checksum validation.

    Pickle is intentionally restricted to application-owned artifacts. This function
    must never be connected to user uploads or automatic downloads.
    """
    metadata_path = artifact_directory / ARTIFACT_METADATA_FILENAME
    if not metadata_path.is_file():
        raise PredictorArtifactNotFoundError("predictor metadata file is missing")
    try:
        raw_metadata: Any = json.loads(metadata_path.read_text())
        if not isinstance(raw_metadata, dict):
            raise TypeError("artifact metadata must be an object")
        if raw_metadata.get("artifact_format_version") == PRODUCTION_ARTIFACT_FORMAT_VERSION:
            metadata = ProductionQualityPredictorArtifactMetadata.model_validate(raw_metadata)
            _validate_production_metadata(metadata)
            expected_checksum = metadata.predictor_sha256
        else:
            metadata = QualityPredictorArtifactMetadata.model_validate(raw_metadata)
            _validate_metadata(metadata)
            expected_checksum = metadata.model_sha256
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise CorruptPredictorArtifactError("predictor metadata is corrupt") from exc
    except (TypeError, ValueError) as exc:
        raise CorruptPredictorArtifactError("predictor metadata is corrupt") from exc
    model_path = artifact_directory / metadata.model_filename
    if not model_path.is_file():
        raise PredictorArtifactNotFoundError("serialized predictor file is missing")
    try:
        model_bytes = model_path.read_bytes()
    except OSError as exc:
        raise CorruptPredictorArtifactError("serialized predictor cannot be read") from exc
    if hashlib.sha256(model_bytes).hexdigest() != expected_checksum:
        raise PredictorArtifactChecksumError("serialized predictor checksum mismatch")
    try:
        pipeline = pickle.loads(model_bytes)
    except Exception as exc:
        raise CorruptPredictorArtifactError("serialized predictor cannot be decoded") from exc
    if not isinstance(pipeline, Pipeline) or not {"preprocess", "classifier"} <= set(
        pipeline.named_steps
    ):
        raise CorruptPredictorArtifactError("serialized object is not a quality pipeline")
    return pipeline, metadata


class SklearnQualityPredictor:
    """Production probability predictor; routing remains a separate Phase 8A concern."""

    def __init__(
        self,
        pipeline: Pipeline,
        metadata: QualityPredictorArtifactMetadata | ProductionQualityPredictorArtifactMetadata,
    ) -> None:
        if isinstance(metadata, ProductionQualityPredictorArtifactMetadata):
            _validate_production_metadata(metadata)
        else:
            _validate_metadata(metadata)
        self._pipeline = pipeline
        self.metadata = metadata
        self._known_candidates = frozenset(metadata.known_candidate_ids)
        self._known_categories = frozenset(metadata.known_categories)

    @classmethod
    def from_trusted_artifact(cls, artifact_directory: Path) -> SklearnQualityPredictor:
        pipeline, metadata = load_trusted_quality_artifact(artifact_directory)
        return cls(pipeline, metadata)

    def predict(
        self,
        request_features: RoutingRequestFeatures,
        candidates: Sequence[ModelConfig],
    ) -> tuple[ModelAcceptabilityPrediction, ...]:
        if request_features.category is None:
            raise MissingRoutingCategoryError(
                "quality prediction requires an explicit routing category"
            )
        if request_features.category not in self._known_categories:
            raise PredictorInputCompatibilityError(
                "request category is not supported by this predictor artifact"
            )
        candidate_ids = [candidate.model_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise PredictorInputCompatibilityError("candidate model IDs must be unique")
        unknown = sorted(set(candidate_ids) - self._known_candidates)
        if unknown:
            raise UnsupportedPredictorCandidateError(
                "candidate is not supported by this predictor artifact: " + ", ".join(unknown)
            )
        canonical = tuple(
            canonical_from_production(request_features, candidate)
            for candidate in candidates
        )
        if not canonical:
            return ()
        probabilities = self._pipeline.predict_proba(canonical_feature_matrix(canonical))[:, 1]
        return tuple(
            ModelAcceptabilityPrediction(
                model_id=candidate.model_id,
                predicted_acceptability=float(probability),
            )
            for candidate, probability in zip(candidates, probabilities)
        )


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ARTIFACT_METADATA_FILENAME",
    "ARTIFACT_MODEL_FILENAME",
    "FOUNDATION_V3_CANDIDATE_IDS",
    "FOUNDATION_V3_PROTOCOL_SHA256",
    "FOUNDATION_V3_SHA256",
    "PHASE_8C0_FORMULATION_EVIDENCE",
    "PRODUCTION_ARTIFACT_FORMAT_VERSION",
    "PRODUCTION_PREDICTOR_SHA256",
    "PRODUCTION_QUALITY_THRESHOLD",
    "ProductionQualityPredictorArtifactMetadata",
    "QualityPredictorArtifactMetadata",
    "SklearnQualityPredictor",
    "expected_metadata",
    "load_trusted_quality_artifact",
    "write_trusted_quality_artifact",
]
