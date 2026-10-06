"""Durable experiment-wide paid-attempt ledgers for the one-time FINAL run."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import UUID, uuid4

from adaptive_llm_gateway.benchmarks.models import BenchmarkResult
from adaptive_llm_gateway.evaluation.models import EvaluationResult


TerminalState = Literal["completed", "failed"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class _Ledger:
    def __init__(self, path: Path, *, experiment_identity: str, kind: str,
                 maximum_calls: int) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.experiment_identity = experiment_identity
        self.kind = kind
        self.maximum_calls = maximum_calls

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any] | None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            value = None
            if self.path.exists():
                value = json.loads(self.path.read_bytes())
                self._validate_header(value)
            yield value

    def _validate_header(self, value: dict[str, Any]) -> None:
        if (value.get("schema_version") != "1.0.0"
                or value.get("kind") != self.kind
                or value.get("experiment_identity") != self.experiment_identity
                or value.get("maximum_calls") != self.maximum_calls
                or not isinstance(value.get("entries"), dict)):
            raise ValueError(f"{self.kind} attempt ledger identity mismatch")

    @staticmethod
    def _ambiguous(value: dict[str, Any]) -> list[str]:
        return sorted(key for key, entry in value["entries"].items()
                      if entry.get("state") == "started")

    def assert_resumable(self) -> None:
        with self._locked() as value:
            if value is None:
                return
            ambiguous = self._ambiguous(value)
            if ambiguous:
                raise ValueError(
                    f"ambiguous started {self.kind} attempt(s) require independent reconciliation: "
                    + ", ".join(ambiguous))

    def counts(self) -> dict[str, int]:
        with self._locked() as value:
            states = (entry["state"] for entry in value["entries"].values()) if value else ()
            result = {state: 0 for state in ("pending", "started", "completed", "failed")}
            for state in states:
                result[state] += 1
            return result


class CandidateAttemptLedger(_Ledger):
    """Exactly-once guard for candidate task/model provider attempts."""

    def __init__(self, path: Path, *, experiment_identity: str,
                 maximum_calls: int = 168) -> None:
        super().__init__(path, experiment_identity=experiment_identity,
                         kind="candidate", maximum_calls=maximum_calls)

    @staticmethod
    def key(task_id: str, model_id: str) -> str:
        return _digest({"task_id": task_id, "candidate_model_id": model_id})

    def initialize(self, *, run_id: UUID, task_bindings: dict[str, dict[str, str]],
                   candidate_ids: tuple[str, ...]) -> None:
        expected = self._expected(task_bindings, candidate_ids)
        if len(expected) != self.maximum_calls:
            raise ValueError("candidate attempt ledger does not contain the frozen call matrix")
        with self._locked() as value:
            if value is None:
                value = {
                    "schema_version": "1.0.0", "kind": self.kind,
                    "experiment_identity": self.experiment_identity,
                    "maximum_calls": self.maximum_calls,
                    "created_at": _now(), "run_ids": [str(run_id)],
                    "entries": {key: {**identity, "state": "pending", "attempt_count": 0}
                                for key, identity in expected.items()},
                }
                _atomic_json(self.path, value)
                return
            self._validate_existing(value, expected, run_id)

    def validate_existing(
        self,
        *,
        run_id: UUID,
        task_bindings: dict[str, dict[str, str]],
        candidate_ids: tuple[str, ...],
    ) -> None:
        """Validate the immutable existing ledger without creating or rewriting it."""
        expected = self._expected(task_bindings, candidate_ids)
        if len(expected) != self.maximum_calls:
            raise ValueError("candidate attempt ledger does not contain the frozen call matrix")
        with self._locked() as value:
            if value is None:
                raise ValueError("candidate attempt ledger is missing")
            self._validate_existing(value, expected, run_id)

    @classmethod
    def _expected(
        cls,
        task_bindings: dict[str, dict[str, str]],
        candidate_ids: tuple[str, ...],
    ) -> dict[str, dict[str, str]]:
        return {
            cls.key(task_id, model_id): {
                "task_id": task_id, "candidate_model_id": model_id,
                "task_binding_sha256": binding["binding_sha256"],
            }
            for task_id, binding in task_bindings.items() for model_id in candidate_ids
        }

    def _validate_existing(
        self,
        value: dict[str, Any],
        expected: dict[str, dict[str, str]],
        run_id: UUID,
    ) -> None:
        actual = {key: {name: entry.get(name) for name in (
            "task_id", "candidate_model_id", "task_binding_sha256")}
            for key, entry in value["entries"].items()}
        if actual != expected:
            raise ValueError("candidate attempt ledger task/model binding mismatch")
        if value.get("run_ids") != [str(run_id)]:
            raise ValueError("candidate attempt ledger run identity mismatch")
        ambiguous = self._ambiguous(value)
        if ambiguous:
            raise ValueError("ambiguous started candidate attempt requires independent reconciliation")

    def reusable(self, task_id: str, model_id: str, run_id: UUID) -> BenchmarkResult | None:
        with self._locked() as value:
            if value is None:
                raise ValueError("candidate attempt ledger is not initialized")
            entry = value["entries"][self.key(task_id, model_id)]
            if entry["state"] == "started":
                raise ValueError("ambiguous started candidate attempt requires reconciliation")
            if entry["state"] == "pending":
                return None
            payload = entry.get("result")
            if not isinstance(payload, dict) or _digest(payload) != entry.get("result_sha256"):
                raise ValueError("terminal candidate ledger result is missing or corrupted")
            result = BenchmarkResult.model_validate(payload)
            if (entry.get("run_id") != str(run_id) or result.run_id != run_id
                    or result.task_id != task_id or result.model_id != model_id):
                raise ValueError("candidate ledger result identity mismatch")
            return result

    def start(self, task_id: str, model_id: str, run_id: UUID, request_id: str) -> None:
        with self._locked() as value:
            if value is None:
                raise ValueError("candidate attempt ledger is not initialized")
            entry = value["entries"][self.key(task_id, model_id)]
            if entry["state"] != "pending" or entry["attempt_count"] != 0:
                raise ValueError("candidate provider attempt is not provably unattempted")
            attempted = sum(item["state"] != "pending" for item in value["entries"].values())
            if attempted >= self.maximum_calls:
                raise ValueError("candidate call budget exhausted")
            entry.update({"state": "started", "attempt_count": 1,
                          "run_id": str(run_id), "request_id": request_id,
                          "started_at": _now()})
            _atomic_json(self.path, value)

    def finish(self, result: BenchmarkResult) -> None:
        with self._locked() as value:
            if value is None:
                raise ValueError("candidate attempt ledger is not initialized")
            entry = value["entries"][self.key(result.task_id, result.model_id)]
            if (entry["state"] != "started" or entry.get("run_id") != str(result.run_id)
                    or entry.get("request_id") != result.request_id):
                raise ValueError("candidate terminal result does not match its started attempt")
            payload = result.model_dump(mode="json")
            entry.update({"state": "completed" if result.success else "failed",
                          "result": payload, "result_sha256": _digest(payload),
                          "finished_at": _now()})
            _atomic_json(self.path, value)


class SemanticAttemptLedger(_Ledger):
    """Exactly-once guard and durable result store for paid semantic judgments."""

    def __init__(self, path: Path, *, experiment_identity: str,
                 evaluator_identity: str, proposition_identity: str,
                 judge_identity: str, maximum_calls: int = 24) -> None:
        super().__init__(path, experiment_identity=experiment_identity,
                         kind="semantic-judge", maximum_calls=maximum_calls)
        self.identities = {
            "evaluator_identity": evaluator_identity,
            "proposition_identity": proposition_identity,
            "judge_identity": judge_identity,
        }

    def _validate_header(self, value: dict[str, Any]) -> None:
        super()._validate_header(value)
        if value.get("identities") != self.identities:
            raise ValueError("semantic-judge attempt ledger identity mismatch")

    def key(self, task_id: str, model_id: str) -> str:
        return _digest({"task_id": task_id, "candidate_model_id": model_id, **self.identities})

    def initialize(self, *, run_id: UUID, pairs: tuple[tuple[str, str], ...]) -> None:
        if len(pairs) > self.maximum_calls or len(set(pairs)) != len(pairs):
            raise ValueError("semantic-judge plan exceeds the frozen call budget")
        expected = {self.key(task, model): (task, model) for task, model in pairs}
        with self._locked() as value:
            if value is None:
                value = {
                    "schema_version": "1.0.0", "kind": self.kind,
                    "experiment_identity": self.experiment_identity,
                    "identities": self.identities, "maximum_calls": self.maximum_calls,
                    "created_at": _now(), "run_id": str(run_id),
                    "entries": {key: {"task_id": pair[0], "candidate_model_id": pair[1],
                                            "state": "pending", "attempt_count": 0}
                                for key, pair in expected.items()},
                }
                _atomic_json(self.path, value)
            else:
                if value.get("run_id") != str(run_id):
                    raise ValueError("semantic-judge ledger belongs to another candidate run")
                actual = {(entry["task_id"], entry["candidate_model_id"])
                          for entry in value["entries"].values()}
                if actual != set(pairs):
                    raise ValueError("semantic-judge plan differs from its durable ledger")
                if self._ambiguous(value):
                    raise ValueError("ambiguous started semantic-judge attempt requires reconciliation")

    def reusable(self, task_id: str, model_id: str) -> EvaluationResult | None:
        with self._locked() as value:
            entry = value["entries"].get(self.key(task_id, model_id)) if value else None
            if entry is None:
                return None
            if entry["state"] == "started":
                raise ValueError("ambiguous started semantic-judge attempt requires reconciliation")
            if entry["state"] == "pending":
                return None
            payload = entry.get("result")
            if not isinstance(payload, dict) or _digest(payload) != entry.get("result_sha256"):
                raise ValueError("terminal semantic-judge result is missing or corrupted")
            return EvaluationResult.model_validate(payload)

    def start(self, task_id: str, model_id: str) -> None:
        with self._locked() as value:
            if value is None:
                raise ValueError("semantic-judge attempt ledger is not initialized")
            entry = value["entries"][self.key(task_id, model_id)]
            if entry["state"] != "pending" or entry["attempt_count"] != 0:
                raise ValueError("semantic-judge attempt is not provably unattempted")
            attempted = sum(item["state"] != "pending" for item in value["entries"].values())
            if attempted >= self.maximum_calls:
                raise ValueError("semantic-judge call budget exhausted")
            entry.update({"state": "started", "attempt_count": 1, "started_at": _now()})
            _atomic_json(self.path, value)

    def finish(self, result: EvaluationResult) -> None:
        with self._locked() as value:
            if value is None:
                raise ValueError("semantic-judge attempt ledger is not initialized")
            entry = value["entries"][self.key(result.task_id, result.model_id)]
            if entry["state"] != "started":
                raise ValueError("semantic-judge result has no matching started attempt")
            payload = result.model_dump(mode="json")
            terminal: TerminalState = "completed" if result.evaluation_status == "evaluated" else "failed"
            entry.update({"state": terminal, "result": payload,
                          "result_sha256": _digest(payload), "finished_at": _now()})
            _atomic_json(self.path, value)
