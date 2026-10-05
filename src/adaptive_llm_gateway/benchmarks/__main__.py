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
    parser.add_argument("--output", type=Path)
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
    parser.add_argument("--authorize-final", action="store_true",
                        help="Confirm independently approved one-time FINAL execution")
    parser.add_argument("--predictor-sha256")
    parser.add_argument("--policy-sha256")
    parser.add_argument(
        "--final-policy", type=Path,
    )
    parser.add_argument(
        "--final-authorization", type=Path,
    )
    parser.add_argument(
        "--final-results-root", type=Path,
    )
    parser.add_argument("--allow-paid", action="store_true", help="Explicitly allow gateway calls that may incur charges")
    args = parser.parse_args()
    final_verification = None
    if args.split == "final":
        if not args.allow_final_evaluation:
            parser.error("Final-test execution requires the explicit final-evaluation gate")
        # Import lazily so ordinary benchmark and test workflows do not load the
        # FINAL harness or its ML dependencies.
        from adaptive_llm_gateway.evaluation.final_harness import (
            CANDIDATES, EXPECTED_REQUESTS, FinalPaths, canonical_final_root,
            sha256, verify_final_freeze,
        )
        from adaptive_llm_gateway.evaluation.final_ledgers import CandidateAttemptLedger
        frozen_paths = FinalPaths()
        canonical_root = canonical_final_root()
        if not args.authorize_final:
            parser.error("Final-test execution requires --authorize-final")
        if args.limit != EXPECTED_REQUESTS or tuple(args.models) != CANDIDATES:
            parser.error("FINAL requires exactly 42 requests and the frozen candidate order")
        if args.protocol is None or args.pricing_readiness is None:
            parser.error("FINAL requires the explicit frozen protocol and pricing readiness")
        try:
            supplied_identities = {
                "dataset_sha256": sha256(args.dataset),
                "split_manifest_sha256": sha256(args.split_manifest),
                "protocol_sha256": sha256(args.protocol),
            }
            if args.pricing_readiness.resolve() != frozen_paths.pricing_readiness.resolve():
                parser.error("FINAL pricing readiness path differs from the reviewed artifact")
            if args.final_policy is not None and args.final_policy.resolve() != frozen_paths.policy.resolve():
                parser.error("FINAL policy path differs from the canonical artifact")
            if (args.final_authorization is not None
                    and args.final_authorization.resolve() != frozen_paths.authorization.resolve()):
                parser.error("FINAL authorization path differs from the canonical tracked artifact")
            if args.output is not None and args.output.resolve() != canonical_root:
                parser.error("FINAL output must equal the canonical experiment root")
            if (args.final_results_root is not None
                    and args.final_results_root.resolve() != canonical_root):
                parser.error("FINAL results root must equal the canonical experiment root")
        except OSError as exc:
            parser.error(str(exc))
        args.output = canonical_root
        args.final_results_root = canonical_root
        try:
            final_verification = verify_final_freeze(
                paths=frozen_paths,
                results_root=canonical_root,
                explicit_authorization=args.authorize_final,
                require_authorization=True,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        if (args.predictor_sha256 is not None
                and args.predictor_sha256 != final_verification.identities["predictor_sha256"]):
            parser.error("caller predictor checksum differs from the verified frozen artifact")
        if (args.policy_sha256 is not None
                and args.policy_sha256 != final_verification.policy_sha256):
            parser.error("caller policy checksum differs from the verified frozen artifact")
        for name, digest in supplied_identities.items():
            if digest != final_verification.identities[name]:
                parser.error(f"caller {name} differs from the verified frozen artifact")
    elif args.output is None:
        args.output = Path("benchmark-results")
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
                    predictor_sha256=(
                        final_verification.identities["predictor_sha256"]
                        if final_verification is not None else args.predictor_sha256),
                    policy_sha256=(
                        final_verification.policy_sha256
                        if final_verification is not None else args.policy_sha256),
                    final_verification=final_verification,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                parser.error(str(exc))
            routing_benchmark_validated = True
            configuration.update({"source_dataset_sha256": source_dataset_sha256,
                                  "execution_split": args.split,
                                  "final_evaluation_explicitly_unlocked": bool(
                                      args.allow_final_evaluation)})
            if final_verification is not None:
                configuration.update({
                    "final_experiment_identity": final_verification.experiment_identity,
                    "final_policy_sha256": final_verification.policy_sha256,
                    "final_authorization_status": final_verification.authorization_status,
                    "final_execution_git_commit": final_verification.git_commit,
                    "final_source_hashes": final_verification.source_hashes,
                    "final_source_set_sha256": final_verification.source_set_sha256,
                })
        execution_models = _model_overrides_from_validated_contract(
            protocol_models=protocol_execution_models,
            routing_benchmark_validated=routing_benchmark_validated,
            selected_model_ids=args.models,
        )
        attempt_ledger = None
        if final_verification is not None:
            attempt_ledger = CandidateAttemptLedger(
                args.final_results_root / ".final-ledgers"
                / final_verification.experiment_identity / "candidate-attempts.json",
                experiment_identity=final_verification.experiment_identity,
            )
        runner = BenchmarkRunner(service, FileBenchmarkRepository(args.output),
                                 configuration=configuration,
                                 output_token_overrides=overrides,
                                 model_configurations=execution_models,
                                 attempt_ledger=attempt_ledger)
        run = asyncio.run(runner.run(dataset, args.models, limit=args.limit))
    except Exception:
        # Avoid emitting connection/provider exception details in a CLI traceback.
        parser.exit(1, "Benchmark failed. Check model IDs, credentials, dataset, and output permissions.\n")
    print(f"Benchmark run: {run.run_id}\nArtifacts: {args.output / str(run.run_id)}")


if __name__ == "__main__":
    main()
