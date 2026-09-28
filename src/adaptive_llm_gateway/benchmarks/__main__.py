"""Explicit opt-in benchmark CLI; defaults to a single task on fake-small."""
import argparse
import asyncio
import json
from pathlib import Path

from adaptive_llm_gateway.bootstrap import configure_gateway, create_development_service
from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings
from .models import load_dataset
from .repository import FileBenchmarkRepository
from .execution_readiness import load_pricing_readiness
from .routing_benchmark_v1 import (
    BENCHMARK_NAME, DEFAULT_PROTOCOL_DIR, ROUTING_BENCHMARK_MODELS,
    SUPPORTED_EXECUTION_PROTOCOLS, load_execution_protocol,
    select_execution_dataset,
)
from .runner import BenchmarkRunner


def _model_overrides_from_validated_contract(
    *,
    protocol_models: tuple[ModelConfig, ...] | None,
    routing_benchmark_validated: bool,
    selected_model_ids: list[str],
) -> dict[str, ModelConfig]:
    """Resolve execution models from the validated source contract.

    Split selection deliberately creates a new dataset object with a descriptive
    name. That derived name is presentation metadata and cannot revoke the
    protocol/model contract already validated against the source dataset.
    """
    if protocol_models is not None:
        configured_models = protocol_models
    elif routing_benchmark_validated:
        configured_models = ROUTING_BENCHMARK_MODELS
    else:
        return {}
    return {
        model.model_id: model
        for model in configured_models
        if model.model_id in selected_model_ids
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/datasets/foundation-v1.json"))
    parser.add_argument("--models", nargs="+", default=["fake-small"])
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--protocol", type=Path,
                        help="Frozen execution protocol containing candidate/task output overrides")
    parser.add_argument(
        "--pricing-readiness", type=Path,
        help="Separate pricing evidence required by protocols that gate paid execution",
    )
    parser.add_argument("--split", choices=("train", "development", "final"),
                        help="Required execution split for Routing Benchmark v1")
    parser.add_argument("--split-manifest", type=Path,
                        default=DEFAULT_PROTOCOL_DIR / "split-manifest.json")
    parser.add_argument("--allow-final-evaluation", action="store_true",
                        help="Explicitly unlock the protected final-test split")
    parser.add_argument("--predictor-sha256")
    parser.add_argument("--policy-sha256")
    parser.add_argument("--allow-paid", action="store_true", help="Explicitly allow gateway calls that may incur charges")
    args = parser.parse_args()
    settings = GatewaySettings.from_environment()
    service = create_development_service()
    configure_gateway(service, settings)
    try:
        models = [service.registry.get(model_id) for model_id in args.models]
        if any(model.provider != "fake" for model in models) and not args.allow_paid:
            parser.error("Real model benchmarks require --allow-paid")
        dataset = load_dataset(args.dataset)
        source_dataset_sha256 = dataset.sha256
        if args.limit <= 0:
            parser.error("--limit must be positive")
        overrides = {}
        protocol_execution_models = None
        routing_benchmark_validated = False
        configuration = {"gateway_timeout_seconds": settings.timeout_seconds}
        if args.protocol is not None:
            try:
                protocol, protocol_execution_models, protocol_sha256 = load_execution_protocol(
                    args.protocol)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                parser.error(str(exc))
            if protocol.get("dataset_sha256") != source_dataset_sha256:
                parser.error("Protocol dataset hash does not match the selected dataset")
            overrides = protocol.get("candidate_task_max_output_tokens", {})
            if not isinstance(overrides, dict):
                parser.error("Protocol output-token overrides are invalid")
            contract = SUPPORTED_EXECUTION_PROTOCOLS[(protocol["protocol"], protocol["version"])]
            if contract.requires_pricing_readiness:
                readiness_path = args.pricing_readiness or args.protocol.with_name(
                    "execution-readiness.json")
                try:
                    readiness = load_pricing_readiness(
                        readiness_path, protocol=protocol,
                        protocol_sha256=protocol_sha256,
                        candidate_models=protocol_execution_models,
                    )
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    parser.error(str(exc))
                configuration["pricing_readiness_version"] = readiness.version
            configuration["execution_protocol_sha256"] = protocol_sha256
        if dataset.name == BENCHMARK_NAME:
            if args.split is None:
                parser.error("Routing Benchmark v1 requires an explicit --split")
            try:
                split_manifest = json.loads(args.split_manifest.read_bytes())
                dataset = select_execution_dataset(
                    dataset, split_manifest, split=args.split,
                    allow_final_evaluation=args.allow_final_evaluation,
                    predictor_sha256=args.predictor_sha256,
                    policy_sha256=args.policy_sha256,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                parser.error(str(exc))
            routing_benchmark_validated = True
            configuration.update({"source_dataset_sha256": source_dataset_sha256,
                                  "execution_split": args.split,
                                  "final_evaluation_explicitly_unlocked": bool(
                                      args.allow_final_evaluation)})
        execution_models = _model_overrides_from_validated_contract(
            protocol_models=protocol_execution_models,
            routing_benchmark_validated=routing_benchmark_validated,
            selected_model_ids=args.models,
        )
        runner = BenchmarkRunner(service, FileBenchmarkRepository(args.output),
                                 configuration=configuration,
                                 output_token_overrides=overrides,
                                 model_configurations=execution_models)
        run = asyncio.run(runner.run(dataset, args.models, limit=args.limit))
    except Exception:
        # Avoid emitting connection/provider exception details in a CLI traceback.
        parser.exit(1, "Benchmark failed. Check model IDs, credentials, dataset, and output permissions.\n")
    print(f"Benchmark run: {run.run_id}\nArtifacts: {args.output / str(run.run_id)}")


if __name__ == "__main__":
    main()
