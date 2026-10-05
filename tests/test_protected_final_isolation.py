import runpy
import tomllib
from pathlib import Path

import adaptive_llm_gateway.benchmarks.models as benchmark_models
import adaptive_llm_gateway.benchmarks.routing_benchmark_v1 as routing_benchmark


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_MODULES = (
    REPOSITORY_ROOT / "tests/test_routing_benchmark_v1.py",
    REPOSITORY_ROOT / "tests/test_hybrid_semantic_evaluator.py",
)


def test_default_pytest_excludes_provider_and_protected_final_tests():
    configuration = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    expression = configuration["tool"]["pytest"]["ini_options"]["addopts"]
    assert "not real_provider" in expression
    assert "not protected_final_data" in expression


def test_real_canonical_fixture_modules_are_explicitly_protected():
    for path in PROTECTED_MODULES:
        source = path.read_text()
        assert "pytestmark = pytest.mark.protected_final_data" in source


def test_importing_protected_test_modules_does_not_materialize_final(monkeypatch):
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("module import attempted to materialize protected FINAL data")

    monkeypatch.setattr(routing_benchmark, "construct_dataset", forbidden)
    monkeypatch.setattr(benchmark_models, "load_dataset", forbidden)
    for path in PROTECTED_MODULES:
        runpy.run_path(str(path), run_name=f"isolation_{path.stem}")
    assert calls == []
