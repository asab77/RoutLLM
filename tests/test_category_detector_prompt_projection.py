import json
from collections import Counter
from pathlib import Path

import pytest

from adaptive_llm_gateway.benchmarks import routing_benchmark_v1 as benchmark


ARTIFACT_ROOT = Path("artifacts/routing-benchmark-v1")


@pytest.mark.parametrize(
    ("split", "expected_total", "expected_per_category"),
    (("train", 140, 20), ("dev", 42, 6)),
)
def test_projection_has_exact_count_balance_and_unique_ids(
    split, expected_total, expected_per_category,
):
    records = benchmark.export_category_detector_prompts(split)

    assert len(records) == expected_total
    assert len({record["task_id"] for record in records}) == expected_total
    assert Counter(record["category"] for record in records) == {
        "classification": expected_per_category,
        "coding": expected_per_category,
        "extraction": expected_per_category,
        "structured_json": expected_per_category,
        "qa": expected_per_category,
        "reasoning": expected_per_category,
        "summarization": expected_per_category,
    }
    assert all(record["split"] == split and record["prompt"] for record in records)


@pytest.mark.parametrize(
    ("split", "manifest_name"),
    (("train", "train-manifest.json"), ("dev", "development-manifest.json")),
)
def test_projection_ids_and_categories_match_existing_split_manifest(split, manifest_name):
    records = benchmark.export_category_detector_prompts(split)
    manifest = json.loads((ARTIFACT_ROOT / manifest_name).read_text(encoding="utf-8"))

    actual = {(record["task_id"], record["category"]) for record in records}
    expected = {(task["task_id"], task["category"]) for task in manifest["tasks"]}
    assert actual == expected


def test_projection_rejects_duplicate_task_ids(tmp_path):
    manifest = json.loads((ARTIFACT_ROOT / "train-manifest.json").read_text(encoding="utf-8"))
    manifest["tasks"][1]["task_id"] = manifest["tasks"][0]["task_id"]
    (tmp_path / "train-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(benchmark.CategoryDetectorProjectionError, match="Duplicate task ID"):
        benchmark.export_category_detector_prompts("train", artifact_root=tmp_path)


def test_final_is_rejected_before_any_family_generator_or_manifest_access(monkeypatch, tmp_path):
    calls = []

    def fail_if_called(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("A prompt family generator was invoked")

    for category in benchmark.CATEGORIES:
        monkeypatch.setitem(benchmark.FAMILY_BUILDERS, category, fail_if_called)
    monkeypatch.setattr(Path, "read_text", fail_if_called)

    with pytest.raises(
        benchmark.CategoryDetectorProjectionError,
        match="permits only 'train' or 'dev'",
    ):
        benchmark.export_category_detector_prompts("final", artifact_root=tmp_path)  # type: ignore[arg-type]

    assert calls == []
