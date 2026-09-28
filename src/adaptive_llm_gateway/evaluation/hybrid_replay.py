"""Offline request preparation for stored successful summary responses."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field

from adaptive_llm_gateway.benchmarks.models import BenchmarkResult, BenchmarkRun
from adaptive_llm_gateway.benchmarks.summarization_spec import PropositionSpecification
from adaptive_llm_gateway.models.schemas import DomainModel

from .hybrid_judge import HybridJudgeRequest

HISTORICAL_REPLAY_RUNS = (
    Path("benchmark-results/4e58c529-f848-47db-ab55-13677b469170"),
    Path("benchmark-results/61707aba-5ab2-4c16-8ec3-eed74555d69c"),
    Path("artifacts/routing-benchmark-v1/pilot-runs/1aa850f6-0862-4704-8f0b-c246c9d990ec"),
)


class HistoricalReplayItem(DomainModel):
    run_id: str
    result_id: str
    task_id: str
    model_id: str
    blind_judge_request: HybridJudgeRequest


def prepare_historical_replay(specification: PropositionSpecification, *,
                              roots: tuple[Path, ...] = HISTORICAL_REPLAY_RUNS
                              ) -> tuple[HistoricalReplayItem, ...]:
    """Build blind judge requests without executing a judge or changing old labels."""
    foundation = {item.canonical_task_id: item for item in specification.tasks
                  if item.specification_id.startswith("foundation-")}
    routing = {item.canonical_task_id: item for item in specification.tasks
               if item.specification_id.startswith("rb12-")}
    replay: list[HistoricalReplayItem] = []
    for root in roots:
        run = BenchmarkRun.model_validate_json((root / "manifest.json").read_bytes())
        task_by_id = {item.task_id: item for item in run.dataset.tasks}
        lookup = foundation if "foundation" in run.dataset.name else routing
        for path in sorted((root / "results").glob("*.json")):
            result = BenchmarkResult.model_validate_json(path.read_bytes())
            task = task_by_id[result.task_id]
            if task.category != "summarization" or not result.success or result.response is None:
                continue
            try:
                task_specification = lookup[result.task_id]
            except KeyError:
                raise ValueError(f"No proposition specification for {result.task_id}") from None
            request = HybridJudgeRequest.from_specification(
                task_specification, result.response.text).model_copy(update={
                    "material_error_contract": specification.material_error_contract})
            replay.append(HistoricalReplayItem(
                run_id=str(run.run_id), result_id=str(result.result_id),
                task_id=result.task_id, model_id=result.model_id,
                blind_judge_request=request,
            ))
    return tuple(replay)
