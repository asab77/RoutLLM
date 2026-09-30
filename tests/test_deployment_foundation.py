import asyncio
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import sklearn
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.api.limits import RequestLimitSettings
from adaptive_llm_gateway.application.adaptive_config import AdaptiveRuntime
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.errors import (
    CorruptPredictorArtifactError,
    IncompatibleFeatureSchemaError,
    PredictorArtifactChecksumError,
)
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse
from adaptive_llm_gateway.providers.gateway_config import CANDIDATE_MODELS
from adaptive_llm_gateway.routing.features import ProductionRequestFeatureExtractor
from adaptive_llm_gateway.routing.policy import CostAwareRoutingPolicy
from adaptive_llm_gateway.routing.predictor import (
    PRODUCTION_PREDICTOR_SHA256,
    PRODUCTION_QUALITY_THRESHOLD,
    ProductionQualityPredictorArtifactMetadata,
    SklearnQualityPredictor,
    load_trusted_quality_artifact,
)
from adaptive_llm_gateway.routing.service import RoutingDecisionService

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ARTIFACT = ROOT / "deploy" / "router"


def _copy_production_artifact(tmp_path: Path) -> Path:
    target = tmp_path / "router"
    shutil.copytree(PRODUCTION_ARTIFACT, target)
    return target


def _rewrite_metadata(directory: Path, update) -> None:
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    update(metadata)
    path.write_text(json.dumps(metadata))


def test_frozen_production_artifact_exists_has_exact_hash_and_loads():
    predictor = PRODUCTION_ARTIFACT / "predictor.pkl"
    assert predictor.is_file()
    assert hashlib.sha256(predictor.read_bytes()).hexdigest() == PRODUCTION_PREDICTOR_SHA256
    pipeline, metadata = load_trusted_quality_artifact(PRODUCTION_ARTIFACT)
    assert set(pipeline.named_steps) == {"preprocess", "classifier"}
    assert isinstance(metadata, ProductionQualityPredictorArtifactMetadata)
    assert metadata.approved_quality_threshold == PRODUCTION_QUALITY_THRESHOLD == 0.80
    assert metadata.runtime_compatibility.scikit_learn_version == sklearn.__version__ == "1.9.1"
    assert metadata.source_validation_metadata.deployment_status == "DEV_VALIDATION_CANDIDATE_NOT_DEPLOYED"


def test_checksum_is_rejected_before_unpickle(monkeypatch, tmp_path):
    directory = _copy_production_artifact(tmp_path)
    predictor = directory / "predictor.pkl"
    predictor.write_bytes(predictor.read_bytes() + b"corrupt")
    unpickle = AsyncMock()
    monkeypatch.setattr(pickle, "loads", unpickle)
    with pytest.raises(PredictorArtifactChecksumError):
        load_trusted_quality_artifact(directory)
    unpickle.assert_not_called()


def test_malformed_production_metadata_is_rejected(tmp_path):
    directory = _copy_production_artifact(tmp_path)
    (directory / "metadata.json").write_text("not-json")
    with pytest.raises(CorruptPredictorArtifactError):
        load_trusted_quality_artifact(directory)


def test_wrong_production_feature_contract_is_rejected(tmp_path):
    directory = _copy_production_artifact(tmp_path)
    _rewrite_metadata(directory, lambda value: value.update(canonical_feature_schema_version="other"))
    with pytest.raises(IncompatibleFeatureSchemaError):
        load_trusted_quality_artifact(directory)


def test_wrong_production_candidate_set_is_rejected(tmp_path):
    directory = _copy_production_artifact(tmp_path)
    _rewrite_metadata(directory, lambda value: value["known_candidate_ids"].pop())
    with pytest.raises(CorruptPredictorArtifactError):
        load_trusted_quality_artifact(directory)


def test_tracked_predictor_routes_synthetic_request_deterministically():
    request = InferenceRequest(
        prompt="Compare the supplied options and return one concise answer.",
        max_output_tokens=160,
        temperature=0,
    )
    extractor = ProductionRequestFeatureExtractor()
    features = extractor.extract(request, category_hint="qa")
    predictor = SklearnQualityPredictor.from_trusted_artifact(PRODUCTION_ARTIFACT)
    predictions = predictor.predict(features, CANDIDATE_MODELS)
    probabilities = {
        item.model_id: item.predicted_acceptability for item in predictions
    }
    assert tuple(probabilities) == tuple(
        candidate.model_id for candidate in CANDIDATE_MODELS
    )
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values())

    service = RoutingDecisionService(
        predictor,
        feature_extractor=extractor,
        policy=CostAwareRoutingPolicy(),
    )
    first = service.route(
        request, CANDIDATE_MODELS, 0.80, category_hint="qa"
    )
    second = service.route(
        request, CANDIDATE_MODELS, 0.80, category_hint="qa"
    )
    assert first == second
    assert first.selected_model_id in probabilities
    assert first.selected_predicted_acceptability == probabilities[first.selected_model_id]
    assert first.threshold_satisfied is (first.qualifying_candidate_count > 0)
    assert first.fallback_used is (not first.threshold_satisfied)
    assert first.quality_threshold == 0.80


def _adaptive_app():
    service = create_development_service()
    generate = AsyncMock()
    response = InferenceResponse(
        text="ok",
        model_id="fake-small",
        provider="fake",
        input_tokens=1,
        output_tokens=1,
        latency_ms=0,
        estimated_cost_usd="0",
    )
    generate.return_value = SimpleNamespace(
        response=response,
        routing_decision=SimpleNamespace(
            selected_model_id="fake-small",
            threshold_satisfied=True,
            fallback_used=False,
            reason="quality_threshold_met",
        ),
        execution=None,
    )
    runtime = AdaptiveRuntime(
        SimpleNamespace(generate=generate),
        ("fake-small",),
        approved_quality_threshold=0.80,
    )
    return create_app(service, runtime), generate


@pytest.mark.parametrize("threshold", [None, 0.80])
def test_production_threshold_omitted_or_exact_is_accepted(threshold):
    app, generate = _adaptive_app()
    payload = {"prompt": "threshold check", "category": "qa"}
    if threshold is not None:
        payload["quality_threshold"] = threshold
    with TestClient(app) as client:
        response = client.post("/v1/inference/adaptive", json=payload)
    assert response.status_code == 200
    assert generate.await_args.kwargs["quality_threshold"] == 0.80


@pytest.mark.parametrize("threshold", [0, 0.5, 0.79, 0.81, 1])
def test_unapproved_production_threshold_is_rejected_before_provider(threshold):
    app, generate = _adaptive_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/inference/adaptive",
            json={"prompt": "threshold check", "category": "qa", "quality_threshold": threshold},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_adaptive_request"
    generate.assert_not_awaited()


def test_internal_routing_policy_still_accepts_nonproduction_thresholds():
    from adaptive_llm_gateway.routing.policy import (
        CandidatePrediction,
        CostAwareRoutingPolicy,
    )

    decision = CostAwareRoutingPolicy().route(
        (CandidatePrediction(model_id="internal", predicted_acceptability=0.75,
                             projected_cost_usd="0.01"),),
        0.73,
    )
    assert decision.quality_threshold == 0.73


@pytest.mark.parametrize(
    "payload",
    [
        {"model_id": "fake-small", "prompt": "x" * 12_001, "max_output_tokens": 1},
        {"model_id": "fake-small", "prompt": "normal", "max_output_tokens": 513},
    ],
)
def test_public_cost_bounds_reject_before_provider(payload):
    service = create_development_service()
    service.generate = AsyncMock()
    with TestClient(create_app(
        service,
        request_limits=RequestLimitSettings(
            max_prompt_characters=12_000, max_output_tokens=512
        ),
    )) as client:
        response = client.post("/v1/inference", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    service.generate.assert_not_awaited()


def test_normal_public_request_remains_valid():
    service = create_development_service()
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/inference",
            json={"model_id": "fake-small", "prompt": "normal demo", "max_output_tokens": 8},
        )
    assert response.status_code == 200


def test_request_limit_settings_have_production_values_and_hard_ceilings(monkeypatch):
    monkeypatch.setenv("ROUTELLM_MAX_PROMPT_CHARACTERS", "12000")
    monkeypatch.setenv("ROUTELLM_MAX_OUTPUT_TOKENS", "512")
    assert RequestLimitSettings.from_environment() == RequestLimitSettings(
        max_prompt_characters=12_000, max_output_tokens=512
    )
    with pytest.raises(ValueError):
        RequestLimitSettings(max_prompt_characters=100_001)
    with pytest.raises(ValueError):
        RequestLimitSettings(max_output_tokens=8_193)


def test_dockerfile_and_build_context_are_production_only():
    dockerfile = (ROOT / "Dockerfile").read_text()
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()
    assert "python:3.12.12-slim-bookworm" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert '"--workers", "1"' in dockerfile
    assert '"--forwarded-allow-ips", "172.30.0.2"' in dockerfile
    assert "--reload" not in dockerfile
    assert "FROM node:24.9.0-alpine AS frontend-build" in dockerfile
    assert "RUN npm ci" in dockerfile
    assert "RUN npm run build" in dockerfile
    assert "FROM caddy:2.10.2-alpine AS edge" in dockerfile
    assert "COPY --from=frontend-build /build/frontend/dist /srv" in dockerfile
    assert "COPY --chown=routellm:routellm deploy/router" in dockerfile
    assert "artifacts" in dockerignore and "benchmark-results" in dockerignore
    assert "frontend/node_modules" in dockerignore and "frontend/dist" in dockerignore
    assert ".env" in dockerignore and "!deploy/router/**" in dockerignore


def _compose_config() -> dict:
    result = subprocess.run(
        [
            "docker", "compose", "--env-file", "deploy/.env.production.example",
            "--profile", "tools", "-f", "compose.prod.yaml", "config", "--format", "json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _domainless_compose_config() -> dict:
    environment = os.environ.copy()
    environment.pop("ROUTELLM_DOMAIN", None)
    environment.pop("ACME_EMAIL", None)
    environment.update(
        {
            "AI_GATEWAY_API_KEY": "placeholder-no-provider-call",
            "POSTGRES_PASSWORD": "placeholder-bootstrap-password",
            "POSTGRES_APP_USER": "routellm",
            "POSTGRES_APP_PASSWORD": "placeholder-app-password",
            "RATE_LIMIT_HMAC_SECRET": "placeholder-rate-secret-at-least-32-characters",
            "METRICS_BEARER_TOKEN": "placeholder-metrics-secret-at-least-32-characters",
            "DEMO_AUTH_USER": "demo",
            "DEMO_AUTH_PASSWORD_HASH": "placeholder-caddy-hash",
        }
    )
    result = subprocess.run(
        [
            "docker", "compose", "--env-file", "/dev/null", "--profile", "tools",
            "-f", "compose.prod.yaml", "-f", "compose.demo.yaml", "config", "--format", "json",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_production_compose_is_private_persistent_and_restartable():
    config = _compose_config()
    services = config["services"]
    assert set(services) == {"app", "caddy", "migrate", "postgres", "redis"}
    assert "ports" not in services["app"]
    assert "ports" not in services["postgres"]
    assert "ports" not in services["redis"]
    assert {item["published"] for item in services["caddy"]["ports"]} == {"80", "443"}
    assert services["app"]["restart"] == "unless-stopped"
    assert services["postgres"]["restart"] == "unless-stopped"
    assert services["redis"]["restart"] == "unless-stopped"
    assert services["caddy"]["restart"] == "unless-stopped"
    assert all("healthcheck" in services[name] for name in ("app", "postgres", "redis"))
    assert services["migrate"]["image"] == services["app"]["image"]
    assert services["app"]["build"]["target"] == "runtime"
    assert services["migrate"]["build"]["target"] == "runtime"
    assert services["caddy"]["build"]["target"] == "edge"
    assert services["migrate"]["command"][-2:] == ["upgrade", "head"]
    assert any("postgres_data" in item["source"] for item in services["postgres"]["volumes"])
    assert not services["redis"].get("volumes")
    assert any("caddy_data" in item["source"] for item in services["caddy"]["volumes"])


def test_domainless_compose_needs_no_domain_or_acme_and_keeps_services_private():
    config = _domainless_compose_config()
    services = config["services"]
    assert set(services) == {"app", "caddy", "migrate", "postgres", "redis"}
    assert "ROUTELLM_DOMAIN" not in services["caddy"].get("environment", {})
    assert "ACME_EMAIL" not in services["caddy"].get("environment", {})
    assert {item["published"] for item in services["caddy"]["ports"]} == {"80"}
    assert all("ports" not in services[name] for name in ("app", "postgres", "redis"))
    caddy_mount = next(
        item for item in services["caddy"]["volumes"]
        if item["target"] == "/etc/caddy/Caddyfile"
    )
    assert caddy_mount["source"].endswith("deploy/Caddyfile.http")


def test_domainless_edge_preserves_auth_blocking_and_proxy_security():
    caddy = (ROOT / "deploy" / "Caddyfile.http").read_text()
    assert caddy.startswith(":80 {")
    assert "{$ROUTELLM_DOMAIN}" not in caddy
    assert "{$ACME_EMAIL}" not in caddy
    assert "basic_auth {" in caddy
    assert "@api path /v1 /v1/*" in caddy
    assert "@probes path /health /ready" in caddy
    assert "root * /srv" in caddy
    assert "try_files {path} /index.html" in caddy
    assert "file_server" in caddy
    for path in ("/metrics", "/v1/metrics/summary", "/v1/benchmarks/*", "/docs", "/redoc", "/openapi.json"):
        assert path in caddy
    assert "respond @protected 404" in caddy
    assert "header_up -X-Forwarded-For" in caddy
    assert "header_up X-Forwarded-For {remote_host}" in caddy


def test_domain_https_edge_remains_configured_for_automatic_tls():
    caddy = (ROOT / "deploy" / "Caddyfile").read_text()
    assert "email {$ACME_EMAIL}" in caddy
    assert "{$ROUTELLM_DOMAIN} {" in caddy
    assert "basic_auth {" in caddy
    assert "@api path /v1 /v1/*" in caddy
    assert "root * /srv" in caddy


def test_edge_policy_gates_paid_and_blocks_operational_routes():
    caddy = (ROOT / "deploy" / "Caddyfile").read_text()
    assert "basic_auth {" in caddy
    assert "@api path /v1 /v1/*" in caddy
    assert "@probes path /health /ready" in caddy
    for path in ("/metrics", "/v1/metrics/summary", "/v1/benchmarks/*", "/docs", "/redoc", "/openapi.json"):
        assert path in caddy
    assert "respond @protected 404" in caddy
    assert all(path not in caddy.split("@protected", 1)[1].splitlines()[0]
               for path in ("/health", "/ready", "/v1/models"))
    assert "header_up -X-Forwarded-For" in caddy
    assert "header_up X-Forwarded-For {remote_host}" in caddy


async def _proxy_client(trusted_peer: str, forwarded: str) -> tuple[str, int]:
    captured = {}

    async def application(scope, receive, send):
        captured["client"] = scope["client"]

    middleware = ProxyHeadersMiddleware(application, trusted_hosts=["172.30.0.2"])
    scope = {
        "type": "http",
        "client": (trusted_peer, 1234),
        "headers": [(b"x-forwarded-for", forwarded.encode())],
        "scheme": "http",
    }
    await middleware(scope, AsyncMock(), AsyncMock())
    return captured["client"]


def test_proxy_trust_is_narrow_distinguishes_clients_and_ignores_forgery():
    first = asyncio.run(_proxy_client("172.30.0.2", "203.0.113.10"))
    second = asyncio.run(_proxy_client("172.30.0.2", "203.0.113.11"))
    forged = asyncio.run(_proxy_client("172.30.0.99", "198.51.100.1"))
    assert first[0] == "203.0.113.10"
    assert second[0] == "203.0.113.11"
    assert first != second
    assert forged == ("172.30.0.99", 1234)


def test_production_environment_is_placeholder_only_and_not_copied():
    template = (ROOT / "deploy" / ".env.production.example").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "REPLACE_WITH" in template
    assert "AI_GATEWAY_API_KEY=REPLACE_WITH" in template
    assert "ROUTELLM_DEFAULT_MODEL_ID=REPLACE_WITH_ENABLED_MODEL_ID" in template
    assert "local_dev_only" not in template
    assert "COPY ." not in dockerfile
    assert ".env.production" not in dockerfile


def test_ci_secret_scan_matches_key_shapes_without_benign_hyphens():
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    expression = (
        r"(sk-[A-Za-z0-9_-]{20,}|vercel_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"
    )
    assert expression in workflow
    pattern = re.compile(expression)
    assert all(
        pattern.search(value) is None
        for value in ("risk-matrix", "task-detection", "sk-short")
    )
    assert all(
        pattern.search(value) is not None
        for value in (
            "sk-" + "abcdefghijklmnopqrstuvwx",
            "vercel_" + "abcdefghijklmnopqrst",
            "AKIA" + "ABCDEFGHIJKLMNOP",
        )
    )
