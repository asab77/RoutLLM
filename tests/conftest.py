import pytest

from adaptive_llm_gateway.models import ModelConfig


@pytest.fixture(autouse=True)
def isolate_database_environment(monkeypatch, request):
    """Normal tests must never write to a developer's configured application DB.

    Integration tests opt in using the separate TEST_DATABASE_URL variable.
    Configuration-specific tests can explicitly set these values themselves.
    """
    if not request.node.get_closest_marker("real_provider"):
        monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("AI_GATEWAY_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("INFERENCE_DEADLINE_SECONDS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_REQUIRED", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("RATE_LIMIT_HMAC_SECRET", raising=False)
    monkeypatch.delenv("REDIS_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_CLIENT_CAPACITY", raising=False)
    monkeypatch.delenv("RATE_LIMIT_CLIENT_WINDOW_SECONDS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_GLOBAL_CAPACITY", raising=False)
    monkeypatch.delenv("RATE_LIMIT_GLOBAL_WINDOW_SECONDS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_EXPLICIT_WEIGHT", raising=False)
    monkeypatch.delenv("RATE_LIMIT_ADAPTIVE_WEIGHT", raising=False)
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    monkeypatch.delenv("METRICS_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TELEMETRY_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("ROUTELLM_ADAPTIVE_ARTIFACT_PATH", raising=False)
    monkeypatch.delenv("ROUTELLM_ADAPTIVE_CANDIDATES", raising=False)
    # Ordinary application tests use the registered offline fake model. Tests
    # for missing chat configuration explicitly delete this variable.
    monkeypatch.setenv("ROUTELLM_DEFAULT_MODEL_ID", "fake-small")
    monkeypatch.delenv("ROUTELLM_MAX_PROMPT_CHARACTERS", raising=False)
    monkeypatch.delenv("ROUTELLM_MAX_OUTPUT_TOKENS", raising=False)


@pytest.fixture
def model() -> ModelConfig:
    return ModelConfig(
        model_id="fake-small",
        provider="fake",
        provider_model_name="echo-v1",
        input_cost_per_1m_tokens="0.15",
        output_cost_per_1m_tokens="0.60",
        context_window=4096,
    )


def pytest_addoption(parser):
    parser.addoption("--run-real-provider", action="store_true", help="Allow explicitly selected paid gateway smoke test")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-real-provider"):
        for item in items:
            if item.get_closest_marker("real_provider"):
                item.add_marker(pytest.mark.skip(reason="Paid test requires --run-real-provider and -m real_provider"))
