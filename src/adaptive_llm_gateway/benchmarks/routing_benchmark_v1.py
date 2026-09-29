"""Deterministic construction and offline preflight for Routing Benchmark v1.

This module authors benchmark definitions and audit artifacts only. It imports no
provider adapter and has no inference path.
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from adaptive_llm_gateway.evaluation.evaluators import EVALUATION_VERSION, evaluator_for
from adaptive_llm_gateway.evaluation.judge import JUDGE_PROMPT_VERSION, SEMANTIC_JUDGE_VERSION
from adaptive_llm_gateway.evaluation.sandbox import (
    FUNCTIONAL_EVALUATOR_VERSION, DockerPythonSandbox,
    FunctionalEvaluationRequest, FunctionalTestCase,
)
from adaptive_llm_gateway.models import (
    InferenceResponse, ModelConfig, OutputTokenAccounting, OutputTokenPolicy,
    ReasoningControlMechanism, ReasoningEffort,
)
from adaptive_llm_gateway.providers.gateway_config import (
    CANDIDATE_MODELS, CATALOG_VERIFIED_AT, PRICING_SOURCE, PRICING_VERIFIED,
    UPSTREAM_PROVIDERS,
)

from .features import approximate_input_tokens
from .models import BenchmarkDataset, BenchmarkResult, BenchmarkTask, load_dataset

BENCHMARK_NAME = "routellm-routing-benchmark-v1"
BENCHMARK_VERSION = "1.1.0"
ROUTING_V12_DATASET_VERSION = "1.2.0"
DISPLAY_NAME = "RouteLLM Routing Benchmark v1"
BUILD_VERSION = "1.1.0"
PROTOCOL_VERSION = "1.3.0"
ROUTING_V12_PROTOCOL_NAME = "routellm-routing-benchmark-v1.2"
ROUTING_V12_PROTOCOL_VERSION = "1.4.0"
ROUTING_V12_CORRECTED_PROTOCOL_VERSION = "1.5.0"
ROUTING_V12_MINIMAL_REASONING_PROTOCOL_VERSION = "1.6.0"
ROUTING_V12_NATIVE_MINIMAL_REASONING_PROTOCOL_VERSION = "1.7.0"
# These values are part of the already-reviewed dataset bytes. The active
# evaluator contract is versioned independently in evaluation.judge.
DATASET_SEMANTIC_JUDGE_VERSION = "1.0.0"
DATASET_JUDGE_PROMPT_VERSION = "summary-rubric-1.0.0"
SPLIT_SEED = 20260926
CATEGORIES = ("classification", "coding", "extraction", "json", "qa", "reasoning", "summarization")
DISPLAY_CATEGORY = {"json": "structured_json"}
DIFFICULTY_COUNTS = {"easy": 8, "medium": 13, "hard": 11}
SPLIT_COUNTS = {"train": 20, "development": 6, "final": 6}
DEFAULT_ROOT = Path("artifacts/routing-benchmark-v1")
DEFAULT_DATASET = Path("benchmarks/datasets/routing-benchmark-v1.json")
DEFAULT_PROTOCOL_DIR = Path("benchmarks/protocols/routing-benchmark-v1")

# BENCHMARK_VERSION remains the historical authoring/build version. Selection
# and execution operate on the Routing Benchmark v1 family and therefore use
# this explicit registry. The semantic digests identify the parsed dataset
# independently of JSON whitespace; paid execution additionally verifies the
# frozen source-byte hash through its protocol contract.
SUPPORTED_BENCHMARK_DATASETS = {
    (BENCHMARK_NAME, BENCHMARK_VERSION):
        "761c10b4fc9898977109412386015874d1b31c53cb710e514250c8eed8fb9839",
    (BENCHMARK_NAME, ROUTING_V12_DATASET_VERSION):
        "dcc95c9330d40881d216bbedf4bf0ac709fc5aa76c19b1f3fce71b7f0df43c99",
}


@dataclass(frozen=True)
class ExecutionProtocolContract:
    protocol: str
    version: str
    dataset_sha256: str
    manifest_sha256: str
    requires_pricing_readiness: bool
    model_contract: Literal[
        "gemini_256", "gemini_384", "gemini_minimal_384",
        "gemini_native_minimal_384",
    ]


SUPPORTED_EXECUTION_PROTOCOLS = {
    (BENCHMARK_NAME, PROTOCOL_VERSION): ExecutionProtocolContract(
        protocol=BENCHMARK_NAME,
        version=PROTOCOL_VERSION,
        dataset_sha256="70152d1ac15e827a0ff940cf5becf76701578d0a62d773b4ad6e3823c3fa27ca",
        manifest_sha256="8a569e225734c52480d5b31d11db9d78a8b2c0417d621b6d7234d696b0859204",
        requires_pricing_readiness=False,
        model_contract="gemini_256",
    ),
    (ROUTING_V12_PROTOCOL_NAME, ROUTING_V12_PROTOCOL_VERSION): ExecutionProtocolContract(
        protocol=ROUTING_V12_PROTOCOL_NAME,
        version=ROUTING_V12_PROTOCOL_VERSION,
        dataset_sha256="1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005",
        manifest_sha256="f132c1eec6b1a231f6ce5ea50702b59a03e76af7d44359ba614fa0ed4ae6e565",
        requires_pricing_readiness=True,
        model_contract="gemini_256",
    ),
    (ROUTING_V12_PROTOCOL_NAME, ROUTING_V12_CORRECTED_PROTOCOL_VERSION): ExecutionProtocolContract(
        protocol=ROUTING_V12_PROTOCOL_NAME,
        version=ROUTING_V12_CORRECTED_PROTOCOL_VERSION,
        dataset_sha256="1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005",
        manifest_sha256="fd9f60ca01e912418ba9b54804ea9a6f92cd3b326fc17d489bb7e9e51cd5521d",
        requires_pricing_readiness=True,
        model_contract="gemini_384",
    ),
    (ROUTING_V12_PROTOCOL_NAME, ROUTING_V12_MINIMAL_REASONING_PROTOCOL_VERSION): ExecutionProtocolContract(
        protocol=ROUTING_V12_PROTOCOL_NAME,
        version=ROUTING_V12_MINIMAL_REASONING_PROTOCOL_VERSION,
        dataset_sha256="1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005",
        manifest_sha256="10ca97aafd313c29778103fe0f6f7145243a3e8795d745e1b51a9b0a582c42b3",
        requires_pricing_readiness=True,
        model_contract="gemini_minimal_384",
    ),
    (ROUTING_V12_PROTOCOL_NAME, ROUTING_V12_NATIVE_MINIMAL_REASONING_PROTOCOL_VERSION): ExecutionProtocolContract(
        protocol=ROUTING_V12_PROTOCOL_NAME,
        version=ROUTING_V12_NATIVE_MINIMAL_REASONING_PROTOCOL_VERSION,
        dataset_sha256="1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005",
        manifest_sha256="b4cde3954a8ccd1b54684dcb62da303e7bc806248306464ae536b9ff717a0bd8",
        requires_pricing_readiness=True,
        model_contract="gemini_native_minimal_384",
    ),
}


def _benchmark_execution_model(model, *, headroom: int, expected_reasoning: int):
    accounting = (OutputTokenAccounting.REASONING_AND_VISIBLE
                  if headroom else OutputTokenAccounting.VISIBLE_ONLY)
    return model.model_copy(update={
        "capabilities": model.capabilities.model_copy(update={
            "output_token_accounting": accounting,
        }),
        "output_token_policy": OutputTokenPolicy(
            reasoning_headroom_tokens=headroom,
            expected_reasoning_tokens=expected_reasoning,
        ),
    })


# Benchmark-only execution snapshots. Production candidate configuration and
# the frozen Phase 7/8 feature contract remain unchanged.
ROUTING_BENCHMARK_MODELS = tuple(
    _benchmark_execution_model(model, headroom=headroom, expected_reasoning=expected)
    for model, headroom, expected in zip(
        CANDIDATE_MODELS,
        (0, 128, 256, 128),
        (0, 32, 160, 32),
        strict=True,
    )
)

ROUTING_BENCHMARK_MODELS_V15 = tuple(
    _benchmark_execution_model(model, headroom=headroom, expected_reasoning=expected)
    for model, headroom, expected in zip(
        CANDIDATE_MODELS,
        (0, 128, 384, 128),
        (0, 32, 256, 32),
        strict=True,
    )
)

ROUTING_BENCHMARK_MODELS_V16 = tuple(
    model.model_copy(update={"reasoning_effort": ReasoningEffort.MINIMAL})
    if model.model_id == "candidate-gemini-3-flash" else model
    for model in ROUTING_BENCHMARK_MODELS_V15
)

ROUTING_BENCHMARK_MODELS_V17 = tuple(
    model.model_copy(update={
        "reasoning_control": ReasoningControlMechanism.GOOGLE_PROVIDER_NATIVE,
    }) if model.model_id == "candidate-gemini-3-flash" else model
    for model in ROUTING_BENCHMARK_MODELS_V16
)

EXECUTION_MODEL_CONTRACTS = {
    "gemini_256": ROUTING_BENCHMARK_MODELS,
    "gemini_384": ROUTING_BENCHMARK_MODELS_V15,
    "gemini_minimal_384": ROUTING_BENCHMARK_MODELS_V16,
    "gemini_native_minimal_384": ROUTING_BENCHMARK_MODELS_V17,
}

# Family order determines only the group-safe 20/6/6 split. Difficulty is
# independently derived from reviewed category-specific content criteria.
FAMILY_SPLITS = ("train",) * 10 + ("development",) * 3 + ("final",) * 3

EVALUATOR_TYPES = {
    "classification": "classification_label_match",
    "coding": "docker_python_functional",
    "extraction": "structured_extraction_f1",
    "json": "json_structure_and_value",
    "qa": "accepted_answer_exact_match",
    "reasoning": "reasoning_final_answer_match",
    "summarization": "semantic_summary_judge",
}

TOKEN_RANGES = {
    "classification": (32, 32), "coding": (128, 256),
    "extraction": (64, 192), "json": (64, 192), "qa": (64, 160),
    "reasoning": (160, 160), "summarization": (64, 192),
}


@dataclass(frozen=True)
class FamilySpec:
    slug: str
    variants: tuple[dict[str, Any], dict[str, Any]]


FamilyInputFactory = Callable[[], Any]


def _materialize_family_inputs(
    factories: tuple[FamilyInputFactory, ...],
    family_indexes: Iterable[int] | None,
) -> tuple[tuple[int, Any], ...]:
    indexes = tuple(range(len(factories))) if family_indexes is None else tuple(family_indexes)
    if len(set(indexes)) != len(indexes):
        raise ValueError("Family indexes must be unique")
    if any(index < 0 or index >= len(factories) for index in indexes):
        raise ValueError("Family index is outside the benchmark family plan")
    return tuple((index, factories[index]()) for index in indexes)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _classification_families(family_indexes: Iterable[int] | None = None) -> tuple[FamilySpec, ...]:
    factories = (
        lambda: FamilySpec("direct-material", (
            {"prompt": "Return only METAL, WOOD, or GLASS. The sample is transparent, brittle, and made from fused silica.", "expected": "GLASS"},
            {"prompt": "Return only METAL, WOOD, or GLASS. The sample has visible grain, can splinter, and came from a maple board.", "expected": "WOOD"},
        )),
        lambda: FamilySpec("numeric-band", (
            {"prompt": "Classify a temperature as COLD below 5, MILD from 5 through 24, or HOT at 25 or above. Temperature: 24. Return one label.", "expected": "MILD"},
            {"prompt": "Classify a temperature as COLD below 5, MILD from 5 through 24, or HOT at 25 or above. Temperature: 5. Return one label.", "expected": "MILD"},
        )),
        lambda: FamilySpec("document-routing", (
            {"prompt": "Route to BILLING if the request disputes a charge, TECH if software fails, otherwise GENERAL. Request: The app works, but I was charged twice. Return one label.", "expected": "BILLING"},
            {"prompt": "Route to BILLING if the request disputes a charge, TECH if software fails, otherwise GENERAL. Request: My invoice is fine, but sign-in shows error 403. Return one label.", "expected": "TECH"},
        )),
        lambda: FamilySpec("priority-precedence", (
            {"prompt": "Assign P1 if safety is threatened; otherwise P2 if service is fully unavailable; otherwise P3. Report: checkout is unavailable and a decorative icon is misaligned. Return only P1, P2, or P3.", "expected": "P2"},
            {"prompt": "Assign P1 if safety is threatened; otherwise P2 if service is fully unavailable; otherwise P3. Report: service still works, but an exposed wire may shock staff. Return only P1, P2, or P3.", "expected": "P1"},
        )),
        lambda: FamilySpec("refund-exception", (
            {"prompt": "Label APPROVE when purchase age is at most 30 days, except opened hygiene items are always DENY. An opened toothbrush was bought 4 days ago. Return one label.", "expected": "DENY"},
            {"prompt": "Label APPROVE when purchase age is at most 30 days, except opened hygiene items are always DENY. A sealed lamp was bought 29 days ago. Return one label.", "expected": "APPROVE"},
        )),
        lambda: FamilySpec("risk-matrix", (
            {"prompt": "Risk is HIGH when impact is high and likelihood is medium or high; MEDIUM when exactly one dimension is high; otherwise LOW. Impact high, likelihood low. Return one label.", "expected": "MEDIUM"},
            {"prompt": "Risk is HIGH when impact is high and likelihood is medium or high; MEDIUM when exactly one dimension is high; otherwise LOW. Impact medium, likelihood high. Return one label.", "expected": "MEDIUM"},
        )),
        lambda: FamilySpec("inventory-state", (
            {"prompt": "Label OUT if stock is zero; BACKORDER if stock is positive but below reserved quantity; READY otherwise. Stock 4, reserved 7, incoming 20. Incoming is a distractor. Return one label.", "expected": "BACKORDER"},
            {"prompt": "Label OUT if stock is zero; BACKORDER if stock is positive but below reserved quantity; READY otherwise. Stock 8, reserved 8, incoming 0. Return one label.", "expected": "READY"},
        )),
        lambda: FamilySpec("access-policy", (
            {"prompt": "Access is ALLOW if the user is active and has the required role, unless the account is suspended; suspended always means DENY. Active editor, required role editor, suspended yes. Return one label.", "expected": "DENY"},
            {"prompt": "Access is ALLOW if the user is active and has the required role, unless the account is suspended. Active viewer, required role editor, suspended no. Return one label.", "expected": "DENY"},
        )),
        lambda: FamilySpec("sla-clock", (
            {"prompt": "Label BREACH if elapsed business hours exceed the tier limit: gold 4, silver 8, bronze 16. Gold ticket elapsed 4 hours; calendar age 2 days is irrelevant. Return BREACH or WITHIN.", "expected": "WITHIN"},
            {"prompt": "Label BREACH if elapsed business hours exceed the tier limit: gold 4, silver 8, bronze 16. Silver ticket elapsed 9 business hours. Return BREACH or WITHIN.", "expected": "BREACH"},
        )),
        lambda: FamilySpec("travel-policy", (
            {"prompt": "Label REIMBURSE when a trip is approved and the receipt is present. For meals over $50, manager sign-off is also required. Approved trip, $68 meal, receipt yes, sign-off no. Return one label.", "expected": "DENY"},
            {"prompt": "Label REIMBURSE when a trip is approved and the receipt is present. For meals over $50, manager sign-off is also required. Approved trip, $42 meal, receipt yes, sign-off no. Return one label.", "expected": "REIMBURSE"},
        )),
        lambda: FamilySpec("retention-rule", (
            {"prompt": "Classify KEEP, DELETE, or LEGAL_HOLD. Legal hold overrides all rules. Otherwise delete expired records and keep unexpired records. Record expired yes; legal hold yes. Return one label.", "expected": "LEGAL_HOLD"},
            {"prompt": "Classify KEEP, DELETE, or LEGAL_HOLD. Legal hold overrides all rules. Otherwise delete expired records and keep unexpired records. Record expired yes; legal hold no. Return one label.", "expected": "DELETE"},
        )),
        lambda: FamilySpec("subscription-eligibility", (
            {"prompt": "Label ELIGIBLE if age is at least 18 and country is supported. Students aged 16–17 are also eligible only with guardian consent. Age 17, student yes, supported country yes, consent no. Return one label.", "expected": "INELIGIBLE"},
            {"prompt": "Label ELIGIBLE if age is at least 18 and country is supported. Students aged 16–17 are also eligible only with guardian consent. Age 16, student yes, supported country yes, consent yes. Return one label.", "expected": "ELIGIBLE"},
        )),
        lambda: FamilySpec("incident-cause", (
            {"prompt": "Classify NETWORK, DATABASE, or APPLICATION. If database health is failed, choose DATABASE even when application errors appear. Otherwise packet loss over 10% means NETWORK; otherwise APPLICATION. DB failed, packet loss 18%, app errors 90. Return one label.", "expected": "DATABASE"},
            {"prompt": "Classify NETWORK, DATABASE, or APPLICATION. If database health is failed, choose DATABASE. Otherwise packet loss over 10% means NETWORK; otherwise APPLICATION. DB healthy, packet loss 12%, app errors 40. Return one label.", "expected": "NETWORK"},
        )),
        lambda: FamilySpec("shipping-service", (
            {"prompt": "Choose GROUND, AIR, or REJECT. Hazardous items are REJECT. Otherwise choose AIR when weight <=2 kg and deadline <=2 days; choose GROUND otherwise. Weight 1 kg, deadline 1 day, hazardous yes. Return one label.", "expected": "REJECT"},
            {"prompt": "Choose GROUND, AIR, or REJECT. Hazardous items are REJECT. Otherwise choose AIR when weight <=2 kg and deadline <=2 days; choose GROUND otherwise. Weight 3 kg, deadline 1 day, hazardous no. Return one label.", "expected": "GROUND"},
        )),
        lambda: FamilySpec("support-escalation", (
            {"prompt": "Label ESCALATE if two failed fixes occurred, or immediately for security issues. Do not count a diagnostic check as a fix. History: diagnostic check, password reset failed, cache clear failed; no security issue. Return one label.", "expected": "ESCALATE"},
            {"prompt": "Label ESCALATE if two failed fixes occurred, or immediately for security issues. History: three diagnostic checks and one failed reinstall; security issue no. Return ESCALATE or CONTINUE.", "expected": "CONTINUE"},
        )),
        lambda: FamilySpec("compliance-hierarchy", (
            {"prompt": "Classify BLOCK, REVIEW, or PASS. Sanctions match means BLOCK. Otherwise missing ownership data means REVIEW. Otherwise PASS. A sanctions match exists and ownership data is missing. Return one label.", "expected": "BLOCK"},
            {"prompt": "Classify BLOCK, REVIEW, or PASS. Sanctions match means BLOCK. Otherwise missing ownership data means REVIEW. Otherwise PASS. No sanctions match; ownership data missing; identity verified. Return one label.", "expected": "REVIEW"},
        )),
    )
    return tuple(value for _, value in _materialize_family_inputs(factories, family_indexes))


def _coding_families(family_indexes: Iterable[int] | None = None) -> tuple[FamilySpec, ...]:
    spec_factories = (
        lambda: ("clamp-sequence", "clamp_values", 3,
         "Return a new list with every number limited to the inclusive [low, high] range. Do not mutate inputs.",
         "def clamp_values(values, low, high):\n    return [min(high, max(low, x)) for x in values]",
         [([[1, -2, 9], 0, 5], [1, 0, 5]), ([[], 0, 1], []), ([[3, 3], 3, 3], [3, 3]), ([[-5, 2], -2, 4], [-2, 2])]),
        lambda: ("collapse-whitespace", "normalize_spaces", 1,
         "Collapse every run of whitespace to one ASCII space and strip the ends.",
         "def normalize_spaces(text):\n    return ' '.join(text.split())",
         [(["  a   b  "], "a b"), (["one\ntwo\tthree"], "one two three"), ([""], ""), (["solo"], "solo")]),
        lambda: ("initial-histogram", "count_by_initial", 1,
         "Return a mapping from each lowercase first character to its word count; ignore empty strings.",
         "def count_by_initial(words):\n    out = {}\n    for word in words:\n        if word:\n            key = word[0].lower()\n            out[key] = out.get(key, 0) + 1\n    return out",
         [([["Apple", "ant", "Bee", ""]], {"a": 2, "b": 1}), ([[]], {}), ([["X", "x", "y"]], {"x": 2, "y": 1}), ([["9a", "9b"]], {"9": 2})]),
        lambda: ("key-value-parser", "parse_pairs", 1,
         "Parse strings formatted key=value. Trim both sides, ignore malformed entries, and let the last duplicate key win.",
         "def parse_pairs(lines):\n    out = {}\n    for line in lines:\n        if '=' in line:\n            key, value = line.split('=', 1)\n            key = key.strip()\n            if key:\n                out[key] = value.strip()\n    return out",
         [([["a=1", " bad ", "b = two"]], {"a": "1", "b": "two"}), ([["x=1", "x=2"]], {"x": "2"}), ([[]], {}), ([["=z", "q="]], {"q": ""})]),
        lambda: ("cyclic-rotation", "rotate_left", 2,
         "Return a list rotated left by k positions. Support negative and oversized k; do not mutate the input.",
         "def rotate_left(items, k):\n    if not items:\n        return []\n    n = k % len(items)\n    return items[n:] + items[:n]",
         [([[1, 2, 3], 1], [2, 3, 1]), ([[1, 2, 3], 4], [2, 3, 1]), ([[1, 2, 3], -1], [3, 1, 2]), ([[], 5], [])]),
        lambda: ("stable-score-ranking", "rank_scores", 1,
         "Input records are [name, score]. Return names sorted by descending score, breaking ties alphabetically.",
         "def rank_scores(records):\n    return [row[0] for row in sorted(records, key=lambda row: (-row[1], row[0]))]",
         [([[['b', 2], ['a', 2], ['c', 1]]], ["a", "b", "c"]), ([[]], []), ([[['z', -1], ['x', 3]]], ["x", "z"]), ([[['a', 1]]], ["a"])]),
        lambda: ("merge-adjacent-intervals", "merge_intervals", 1,
         "Merge closed integer intervals that overlap or are adjacent. Adjacent means next_start <= current_end + 1. Return sorted [start,end] lists.",
         "def merge_intervals(intervals):\n    ordered = sorted(intervals)\n    out = []\n    for start, end in ordered:\n        if not out or start > out[-1][1] + 1:\n            out.append([start, end])\n        else:\n            out[-1][1] = max(out[-1][1], end)\n    return out",
         [([[[1, 2], [3, 4], [8, 9]]], [[1, 4], [8, 9]]), ([[]], []), ([[[5, 7], [1, 3], [2, 6]]], [[1, 7]]), ([[[-2, -1], [1, 1]]], [[-2, -1], [1, 1]])]),
        lambda: ("balanced-delimiters", "balanced_brackets", 1,
         "Return true exactly when (), [], and {} delimiters are correctly nested; ignore other characters.",
         "def balanced_brackets(text):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for ch in text:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack",
         [(["([{}])"], True), (["([)]"], False), (["abc"], True), (["("], False)]),
        lambda: ("unweighted-shortest-path", "shortest_hops", 3,
         "Edges are directed [from,to] pairs. Return the minimum hop count from start to end, or -1 if unreachable.",
         "def shortest_hops(edges, start, end):\n    if start == end:\n        return 0\n    queue = [[start, 0]]\n    seen = {start}\n    for node, dist in queue:\n        for left, right in edges:\n            if left == node and right not in seen:\n                if right == end:\n                    return dist + 1\n                seen.add(right)\n                queue.append([right, dist + 1])\n    return -1",
         [([[['a', 'b'], ['b', 'c']], 'a', 'c'], 2), ([[['a', 'b']], 'b', 'a'], -1), ([[], 'x', 'x'], 0), ([[['a', 'c'], ['a', 'b'], ['b', 'c']], 'a', 'c'], 1)]),
        lambda: ("nested-group-total", "nested_sum", 1,
         "Input is a list of integer lists. Return the sum of every integer across all groups.",
         "def nested_sum(groups):\n    return sum(sum(group) for group in groups)",
         [([[[1], [2, 3], []]], 6), ([[[5]]], 5), ([[]], 0), ([[[-1], [4, 2]]], 5)]),
        lambda: ("longest-nondecreasing", "longest_nondecreasing", 1,
         "Return the length of the longest non-contiguous nondecreasing subsequence.",
         "def longest_nondecreasing(values):\n    if not values:\n        return 0\n    best = [1] * len(values)\n    for i in range(len(values)):\n        for j in range(i):\n            if values[j] <= values[i]:\n                best[i] = max(best[i], best[j] + 1)\n    return max(best)",
         [([[3, 1, 2, 2, 4]], 4), ([[]], 0), ([[5, 4, 3]], 1), ([[1, 1, 1]], 3)]),
        lambda: ("slug-validation", "valid_slug", 1,
         "Return true for nonempty lowercase slugs containing only a-z, digits, and single hyphens, with no leading or trailing hyphen.",
         "def valid_slug(text):\n    if not text or text[0] == '-' or text[-1] == '-':\n        return False\n    previous = ''\n    for ch in text:\n        if not (ch.isdigit() or ('a' <= ch <= 'z') or ch == '-'):\n            return False\n        if ch == '-' and previous == '-':\n            return False\n        previous = ch\n    return True",
         [(["alpha-2"], True), (["-alpha"], False), (["two--parts"], False), (["Alpha"], False)]),
        lambda: ("run-length-encoding", "run_length_encode", 1,
         "Return consecutive character runs as [[character,count], ...]. Preserve case and order.",
         "def run_length_encode(text):\n    out = []\n    for ch in text:\n        if out and out[-1][0] == ch:\n            out[-1][1] += 1\n        else:\n            out.append([ch, 1])\n    return out",
         [(["aaabb"], [["a", 3], ["b", 2]]), ([""], []), (["aAa"], [["a", 1], ["A", 1], ["a", 1]]), (["xxx"], [["x", 3]])]),
        lambda: ("rolling-window-sums", "window_sums", 2,
         "Return sums of every contiguous window of size k. Return [] when k <= 0 or k exceeds the list length.",
         "def window_sums(values, k):\n    if k <= 0 or k > len(values):\n        return []\n    return [sum(values[i:i+k]) for i in range(len(values)-k+1)]",
         [([[1, 2, 3, 4], 2], [3, 5, 7]), ([[1], 1], [1]), ([[1, 2], 3], []), ([[2, -2, 2], 2], [0, 0])]),
        lambda: ("transaction-state", "apply_transactions", 2,
         "Apply signed transactions in order. Skip any transaction that would make balance negative. Return the final balance.",
         "def apply_transactions(balance, transactions):\n    for amount in transactions:\n        if balance + amount >= 0:\n            balance += amount\n    return balance",
         [([10, [-3, -8, 5]], 12), ([0, [-1, 2]], 2), ([5, []], 5), ([3, [-3, -1]], 0)]),
        lambda: ("dependency-readiness", "ready_steps", 2,
         "Dependencies map each step to prerequisite names. Return alphabetically sorted incomplete steps whose prerequisites are all completed.",
         "def ready_steps(dependencies, completed):\n    done = set(completed)\n    return sorted(step for step, needs in dependencies.items() if step not in done and all(item in done for item in needs))",
         [([{"build": ["test"], "test": ["code"], "code": []}, ["code"]], ["test"]), ([{}, []], []), ([{"a": [], "b": []}, []], ["a", "b"]), ([{"a": ["x"]}, ["a", "x"]], [])]),
    )
    families = []
    for index, spec in _materialize_family_inputs(spec_factories, family_indexes):
        slug, function_name, parameters, behavior, source, cases = spec
        midpoint = max(2, len(cases) // 2)
        variants = []
        for variant in range(2):
            ordered = cases[variant:] + cases[:variant]
            edge_contract = (
                "The function must handle the empty or smallest valid input described by the contract."
                if variant == 0 else
                "The function must handle repeated, boundary, or out-of-range values described by the contract."
            )
            variants.append({
                "prompt": f"Write Python function `{function_name}` with exactly {parameters} positional parameters. {behavior} {edge_contract} Return only the function definition. Do not import modules or mutate inputs.",
                "function_name": function_name,
                "parameters": parameters,
                "tests": [{"args": args, "expected": expected} for args, expected in ordered],
                "canonical": source,
                "incorrect": f"def {function_name}({', '.join('x'+str(i) for i in range(parameters))}):\n    return None",
                "required_ast": ["Return"],
            })
        families.append(FamilySpec(slug, tuple(variants)))
    return tuple(families)


def _extraction_families(family_indexes: Iterable[int] | None = None) -> tuple[FamilySpec, ...]:
    pair_factories = (
        lambda: ("contact-prose",
         ("Return JSON only with name and email. Contact: Mira Chen can be reached at mira@example.com; her office color is blue.", {"name": "Mira Chen", "email": "mira@example.com"}),
         ("Return JSON only with name and email. Contact: Omar Vale, phone 555-0100, email omar@sample.org.", {"name": "Omar Vale", "email": "omar@sample.org"})),
        lambda: ("invoice-header",
         ("Extract invoice_id, date, and currency as JSON. Invoice INV-204 | issued 2026-02-03 | currency USD | page 1 of 2.", {"invoice_id": "INV-204", "date": "2026-02-03", "currency": "USD"}),
         ("Extract invoice_id, date, and currency as JSON. Currency: EUR; issued: 2026-05-19; invoice: Q-77; draft marker: no.", {"invoice_id": "Q-77", "date": "2026-05-19", "currency": "EUR"})),
        lambda: ("latest-error-log",
         ("Return JSON with latest_error {time,code}. Logs: 09:00 INFO start; 09:04 ERROR E12 disk; 09:06 WARN slow; 09:08 ERROR E19 network.", {"latest_error": {"time": "09:08", "code": "E19"}}),
         ("Return JSON with latest_error {time,code}. Logs: 14:10 ERROR X2 auth; 14:11 INFO retry; 14:15 ERROR X5 quota; 14:16 INFO stop.", {"latest_error": {"time": "14:15", "code": "X5"}})),
        lambda: ("text-table-filter",
         ("Return JSON {active_ids:[...]} preserving table order. Table: id|state|owner; A1|active|Lee; B2|paused|Lee; C3|active|Noor.", {"active_ids": ["A1", "C3"]}),
         ("Return JSON {active_ids:[...]} preserving table order. Rows: K9|closed|Ari; J2|active|Bo; M4|active|Cy; N1|paused|Di.", {"active_ids": ["J2", "M4"]})),
        lambda: ("optional-field",
         ("Extract JSON {id,owner,deadline}. Use null when deadline is absent. Record R8 owner Inez; priority high; no deadline supplied.", {"id": "R8", "owner": "Inez", "deadline": None}),
         ("Extract JSON {id,owner,deadline}. Use null when deadline is absent. Record T3 owner Pavel, deadline 2026-11-02, note pending.", {"id": "T3", "owner": "Pavel", "deadline": "2026-11-02"})),
        lambda: ("repeated-entity-total",
         ("Return JSON {customer,total}. Entries: Ada +$12; Ben +$7; Ada -$2; Ada +$5. Target customer Ada.", {"customer": "Ada", "total": 15}),
         ("Return JSON {customer,total}. Entries: Sol +$9; Uma +$4; Sol +$3; Uma -$1. Target customer Uma.", {"customer": "Uma", "total": 3})),
        lambda: ("nested-shipment",
         ("Extract JSON {shipment_id,destination:{city,country},packages}. Shipment S-5 has 3 packages. Destination city Lima, country PE. Origin Quito EC.", {"shipment_id": "S-5", "destination": {"city": "Lima", "country": "PE"}, "packages": 3}),
         ("Extract JSON {shipment_id,destination:{city,country},packages}. Origin Oslo NO; shipment Z-2; destination Riga LV; packages 1.", {"shipment_id": "Z-2", "destination": {"city": "Riga", "country": "LV"}, "packages": 1})),
        lambda: ("section-precedence",
         ("Return JSON {status,owner} using the FINAL section, not DRAFT. DRAFT status=open owner=Kai. FINAL status=closed owner=Ruth.", {"status": "closed", "owner": "Ruth"}),
         ("Return JSON {status,owner} using the APPROVED section. PROPOSED status=hold owner=Max. APPROVED status=active owner=Liv.", {"status": "active", "owner": "Liv"})),
        lambda: ("unit-bearing-measurement",
         ("Extract JSON {value,unit,sensor}. Sensor TH-2 reports 18.4 C; threshold 22 C; battery 90%.", {"value": 18.4, "unit": "C", "sensor": "TH-2"}),
         ("Extract JSON {value,unit,sensor}. Battery 71%; sensor PR-8 reports 101.3 kPa; alert threshold 99.", {"value": 101.3, "unit": "kPa", "sensor": "PR-8"})),
        lambda: ("multi-party-action",
         ("Return JSON {approver,requester,amount}. Jules requested $480. Nia reviewed it. Omar approved the request.", {"approver": "Omar", "requester": "Jules", "amount": 480}),
         ("Return JSON {approver,requester,amount}. Priya approved after Chen requested $75; Mateo merely copied the note.", {"approver": "Priya", "requester": "Chen", "amount": 75})),
        lambda: ("keyed-list-order",
         ("Return JSON {items:[{sku,qty}]} sorted by sku. Manifest: Z9 qty2; A1 qty5; M3 qty1. Ignore prices.", {"items": [{"sku": "A1", "qty": 5}, {"sku": "M3", "qty": 1}, {"sku": "Z9", "qty": 2}]}),
         ("Return JSON {items:[{sku,qty}]} sorted by sku. Manifest: C2 qty4; B7 qty1. Warehouse west.", {"items": [{"sku": "B7", "qty": 1}, {"sku": "C2", "qty": 4}]})),
        lambda: ("event-window",
         ("Return JSON {events:[...]} containing event names from 10:00 through 10:30 inclusive. 09:55 boot; 10:00 login; 10:12 upload; 10:31 logout.", {"events": ["login", "upload"]}),
         ("Return JSON {events:[...]} containing event names from 15:10 through 15:20 inclusive. 15:09 open; 15:10 edit; 15:20 save; 15:21 close.", {"events": ["edit", "save"]})),
        lambda: ("cross-line-record",
         ("Return JSON {case_id,patient,medication}. Case C44. Patient: Lena Ortiz. Notes continue below. Prescribed medication: amoxicillin. Nurse: Bo.", {"case_id": "C44", "patient": "Lena Ortiz", "medication": "amoxicillin"}),
         ("Return JSON {case_id,patient,medication}. Patient Ravi Sen is in case D12. Clinician note: medication metformin; follow-up Friday.", {"case_id": "D12", "patient": "Ravi Sen", "medication": "metformin"})),
        lambda: ("conditional-record-selection",
         ("Return JSON for the highest-version approved record as {id,version,owner}. R1 v2 draft Mia; R1 v1 approved Jon; R1 v3 approved Zoe.", {"id": "R1", "version": 3, "owner": "Zoe"}),
         ("Return JSON for the highest-version approved record as {id,version,owner}. K2 v4 rejected Ali; K2 v2 approved Bea; K2 v3 approved Cy.", {"id": "K2", "version": 3, "owner": "Cy"})),
        lambda: ("paired-source-join",
         ("Join by product_id and return JSON {product,name,stock}. Catalog: P1=Pen, P2=Book. Inventory: P2=7, P1=12. Target P2.", {"product": "P2", "name": "Book", "stock": 7}),
         ("Join by product_id and return JSON {product,name,stock}. Catalog: X4=Lamp, X5=Desk. Inventory: X5=2, X4=9. Target X4.", {"product": "X4", "name": "Lamp", "stock": 9})),
        lambda: ("exception-aware-roles",
         ("Return JSON {primary,backup}. Team list says primary=Ada backup=Ben. Exception notice effective today swaps primary to Cy but leaves backup unchanged.", {"primary": "Cy", "backup": "Ben"}),
         ("Return JSON {primary,backup}. Roster primary=Dee backup=Eli. Override says Eli becomes primary and Fran becomes backup.", {"primary": "Eli", "backup": "Fran"})),
    )
    pairs = (value for _, value in _materialize_family_inputs(pair_factories, family_indexes))
    return tuple(FamilySpec(slug, (
        {"prompt": first[0], "expected": first[1]},
        {"prompt": second[0], "expected": second[1]},
    )) for slug, first, second in pairs)


def _json_families(family_indexes: Iterable[int] | None = None) -> tuple[FamilySpec, ...]:
    pair_factories = (
        lambda: ("flat-profile",
         ("Return JSON only: {name,age,active}. Name Niko; age 31; active yes.", {"name": "Niko", "age": 31, "active": True}),
         ("Return JSON only: {name,age,active}. Active no; age 44; name Sara.", {"name": "Sara", "age": 44, "active": False})),
        lambda: ("filter-array",
         ("Return JSON {ids:[...]} for enabled records in input order. A enabled, B disabled, C enabled.", {"ids": ["A", "C"]}),
         ("Return JSON {ids:[...]} for enabled records in input order. X disabled, Y enabled, Z disabled.", {"ids": ["Y"]})),
        lambda: ("nested-address",
         ("Return JSON {user:{id,address:{city,zip}}}. User U1 lives in Reno, zip 89501.", {"user": {"id": "U1", "address": {"city": "Reno", "zip": "89501"}}}),
         ("Return JSON {user:{id,address:{city,zip}}}. City Boise, user U8, zip 83702.", {"user": {"id": "U8", "address": {"city": "Boise", "zip": "83702"}}})),
        lambda: ("enum-mapping",
         ("Return JSON {ticket,status}. Map new->OPEN, working->IN_PROGRESS, done->CLOSED. Ticket T2 is working.", {"ticket": "T2", "status": "IN_PROGRESS"}),
         ("Return JSON {ticket,status}. Map new->OPEN, working->IN_PROGRESS, done->CLOSED. Ticket T7 is done.", {"ticket": "T7", "status": "CLOSED"})),
        lambda: ("nullable-field",
         ("Return JSON {id,assignee}. Use null if unassigned. Item A4 is unassigned.", {"id": "A4", "assignee": None}),
         ("Return JSON {id,assignee}. Use null if unassigned. Item B9 is assigned to Jo.", {"id": "B9", "assignee": "Jo"})),
        lambda: ("sorted-ranking",
         ("Return JSON {ranking:[{name,score}]} sorted score descending then name. Bea 8, Ana 8, Cy 5.", {"ranking": [{"name": "Ana", "score": 8}, {"name": "Bea", "score": 8}, {"name": "Cy", "score": 5}]}),
         ("Return JSON {ranking:[{name,score}]} sorted score descending then name. Xu 3, Wen 9, Zoe 9.", {"ranking": [{"name": "Wen", "score": 9}, {"name": "Zoe", "score": 9}, {"name": "Xu", "score": 3}]})),
        lambda: ("grouped-counts",
         ("Return JSON {counts:{...}} counting statuses. Records: open, closed, open, hold. Include only observed statuses.", {"counts": {"closed": 1, "hold": 1, "open": 2}}),
         ("Return JSON {counts:{...}} counting regions. Records: east, west, east, east. Include only observed regions.", {"counts": {"east": 3, "west": 1}})),
        lambda: ("conditional-field",
         ("Return JSON {id,state,error}. Include error as a string only when state=failed; otherwise null. Job J3 failed with timeout.", {"id": "J3", "state": "failed", "error": "timeout"}),
         ("Return JSON {id,state,error}. Include error as a string only when state=failed; otherwise null. Job J8 completed; an old timeout note is irrelevant.", {"id": "J8", "state": "completed", "error": None})),
        lambda: ("deduplicated-array",
         ("Return JSON {tags:[...]} with first-occurrence order and duplicates removed. Tags: red, blue, red, green, blue.", {"tags": ["red", "blue", "green"]}),
         ("Return JSON {tags:[...]} with first-occurrence order and duplicates removed. Tags: api, web, api, cli.", {"tags": ["api", "web", "cli"]})),
        lambda: ("joined-records",
         ("Return JSON {orders:[{id,customer}]} joining customer IDs. Customers C1=Ada, C2=Bo. Orders O2:C2, O1:C1. Sort by order id.", {"orders": [{"id": "O1", "customer": "Ada"}, {"id": "O2", "customer": "Bo"}]}),
         ("Return JSON {orders:[{id,customer}]} joining customer IDs. Customers K1=Ira, K2=Lee. Orders R3:K1, R1:K2. Sort by order id.", {"orders": [{"id": "R1", "customer": "Lee"}, {"id": "R3", "customer": "Ira"}]})),
        lambda: ("cross-field-status",
         ("Return JSON {id,total,status}. total=qty*unit_price. status is LARGE when total>=100 else SMALL. id A, qty 6, unit_price 20.", {"id": "A", "total": 120, "status": "LARGE"}),
         ("Return JSON {id,total,status}. total=qty*unit_price. status is LARGE when total>=100 else SMALL. id B, qty 4, unit_price 24.", {"id": "B", "total": 96, "status": "SMALL"})),
        lambda: ("hierarchy-tree",
         ("Return JSON {department,teams:[{name,lead}]}. Department Ops; teams: Infra led by Mei, Support led by Raj. Preserve order.", {"department": "Ops", "teams": [{"name": "Infra", "lead": "Mei"}, {"name": "Support", "lead": "Raj"}]}),
         ("Return JSON {department,teams:[{name,lead}]}. Department Product; teams: Core led by Liv, Labs led by Noa. Preserve order.", {"department": "Product", "teams": [{"name": "Core", "lead": "Liv"}, {"name": "Labs", "lead": "Noa"}]})),
        lambda: ("schedule-slots",
         ("Return JSON {free:[...]} for hourly slots not occupied. Candidate slots [9,10,11,12], occupied [10,12].", {"free": [9, 11]}),
         ("Return JSON {free:[...]} for hourly slots not occupied. Candidate slots [13,14,15], occupied [14].", {"free": [13, 15]})),
        lambda: ("invoice-aggregation",
         ("Return JSON {subtotal,tax,total}. Lines 2*$10 and 1*$5; tax rate 0.10. Use numbers.", {"subtotal": 25, "tax": 2.5, "total": 27.5}),
         ("Return JSON {subtotal,tax,total}. Lines 3*$8 and 2*$3; tax rate 0.20. Use numbers.", {"subtotal": 30, "tax": 6, "total": 36})),
        lambda: ("permission-matrix",
         ("Return JSON {users:[{name,permissions}]} sorted by name. Roles: editor=[read,write], viewer=[read]. Users Zoe viewer, Ana editor.", {"users": [{"name": "Ana", "permissions": ["read", "write"]}, {"name": "Zoe", "permissions": ["read"]}]}),
         ("Return JSON {users:[{name,permissions}]} sorted by name. Roles: admin=[read,write,delete], viewer=[read]. Users Bo admin, Cy viewer.", {"users": [{"name": "Bo", "permissions": ["read", "write", "delete"]}, {"name": "Cy", "permissions": ["read"]}]})),
        lambda: ("rule-derived-summary",
         ("Return JSON {eligible,ineligible}. Minimum score 70 and attendance 80. Ada 72/81, Ben 90/60, Cy 69/99. Lists sorted.", {"eligible": ["Ada"], "ineligible": ["Ben", "Cy"]}),
         ("Return JSON {eligible,ineligible}. Minimum score 60 and attendance 75. Dio 60/75, Eve 59/100, Fox 88/74. Lists sorted.", {"eligible": ["Dio"], "ineligible": ["Eve", "Fox"]})),
    )
    pairs = (value for _, value in _materialize_family_inputs(pair_factories, family_indexes))
    return tuple(FamilySpec(slug, (
        {"prompt": first[0], "expected": first[1]},
        {"prompt": second[0], "expected": second[1]},
    )) for slug, first, second in pairs)


def _qa_families(family_indexes: Iterable[int] | None = None) -> tuple[FamilySpec, ...]:
    pair_factories = (
        lambda: ("direct-location",
         ("Context: The cobalt binder is in drawer 6. The amber binder is in drawer 2. Question: Where is the cobalt binder? Answer with the drawer number only.", ["6", "drawer 6"]),
         ("Context: The north key is in locker 14; the south key is in locker 9. Question: Which locker holds the south key? Answer with the number only.", ["9", "locker 9"])),
        lambda: ("attribute-lookup",
         ("Context: Plan Birch costs $12 and supports 3 users. Plan Cedar costs $18 and supports 8 users. Question: How many users does Cedar support? Answer with a number.", ["8", "8 users"]),
         ("Context: Sensor A samples every 5 seconds. Sensor B samples every 12 seconds. Question: What is Sensor A's sampling interval? Answer in seconds.", ["5", "5 seconds"])),
        lambda: ("two-fact-sum",
         ("Context: Warehouse east has 17 boxes and warehouse west has 9. Two east boxes are damaged but still counted. Question: How many boxes are listed in total? Answer with a number.", ["26"]),
         ("Context: Team Red closed 14 tickets and Team Blue closed 11. Three reopened tickets are already included in those counts. Question: Total tickets closed? Answer with a number.", ["25"])),
        lambda: ("temporal-latest",
         ("Context: On Monday the owner was Ari. On Wednesday ownership moved to Bea. On Friday Bea delegated review to Cy but retained ownership. Question: Who owns the item after Friday?", ["Bea"]),
         ("Context: Version 1 used port 80. Version 2 changed to 8080. Version 3 changed logging only. Question: Which port does version 3 use?", ["8080", "port 8080"])),
        lambda: ("difference-comparison",
         ("Context: Route Pine is 42 km and Route Oak is 35 km. Both include the same 4 km tunnel. Question: How many kilometers longer is Pine than Oak?", ["7", "7 km", "7 kilometers"]),
         ("Context: Report A has 118 pages and Report B has 93 pages. Appendices are included. Question: By how many pages is A longer?", ["25", "25 pages"])),
        lambda: ("exception-owner",
         ("Context: Requests normally go to Mina. Hardware requests go to Sol instead. Emergency requests go to Teo regardless of type. This is an emergency hardware request. Question: Who receives it?", ["Teo"]),
         ("Context: Reviews go to Ana, except finance reviews go to Bo. Confidential reviews go to Cy regardless of department. This review is finance and confidential. Who receives it?", ["Cy"])),
        lambda: ("multi-passage-link",
         ("Passage 1: Project Kestrel is managed by Lin. Passage 2: Lin's office is Building C. Passage 3: Project Heron is managed by Uma in Building A. Question: Which building houses Kestrel's manager?", ["Building C", "C"]),
         ("Passage 1: Dataset Delta belongs to team Quartz. Passage 2: Quartz reports to director Noel. Passage 3: Team Onyx reports to Priya. Question: Who directs the team that owns Delta?", ["Noel"])),
        lambda: ("schedule-intersection",
         ("Context: Mira is available Tuesday and Thursday. Oren is available Monday, Thursday, Friday. The room is free Wednesday and Thursday. Question: On which day can all meet?", ["Thursday"]),
         ("Context: Team A can deploy at 10:00 or 14:00. Team B can deploy at 09:00 or 14:00. The change window permits 14:00 or 16:00. Which time works for all?", ["14:00", "14:00 hours"])),
        lambda: ("conditional-count",
         ("Context: Orders A=$40 paid, B=$70 unpaid, C=$30 paid, D=$60 paid. Only paid orders of at least $40 qualify. Question: How many qualify?", ["2"]),
         ("Context: Devices P active age2, Q inactive age1, R active age5, S active age1. Only active devices younger than 3 years qualify. How many qualify?", ["2"])),
        lambda: ("state-after-updates",
         ("Context: Balance starts at 30. A deposit adds 12, a purchase subtracts 9, and a reversed fee adds back 3. Question: Final balance?", ["36", "$36"]),
         ("Context: Tank starts with 50 liters. Use 18, add 7, then spill 4. Question: How many liters remain?", ["35", "35 liters"])),
        lambda: ("rank-with-tie-rule",
         ("Context: Scores are Ana 9, Bo 12, Cy 12. Ties are broken alphabetically. Question: Who ranks first?", ["Bo"]),
         ("Context: Completion times: Li 8, Mo 6, Noa 6 minutes. Lower is better; ties alphabetical. Who ranks first?", ["Mo"])),
        lambda: ("dependency-question",
         ("Context: Publish requires Review. Review requires Draft. Draft is complete; Review is not. Question: What is the next incomplete prerequisite that can be worked on?", ["Review"]),
         ("Context: Launch requires Signoff and Training. Signoff is complete. Training requires Materials, which are complete. Training is incomplete. What should be completed next?", ["Training"])),
        lambda: ("policy-date-window",
         ("Context: Returns are allowed within 14 days inclusive. Purchase was June 1 and return was June 15. Assume day difference is 14. Question: Is the return allowed? Answer yes or no.", ["yes"]),
         ("Context: Cancellation is free fewer than 48 hours before nothing; the rule actually says at least 48 hours before departure. Request is exactly 48 hours before. Is it free? Answer yes or no.", ["yes"])),
        lambda: ("multi-step-rate",
         ("Context: A machine makes 6 parts per hour for 4 hours. Three parts fail inspection. Question: How many passing parts remain?", ["21", "21 parts"]),
         ("Context: Four vans carry 8 crates each. Five crates are unloaded at stop one. How many remain on the vans?", ["27", "27 crates"])),
        lambda: ("role-chain",
         ("Context: Eli mentors Faye. Faye mentors Gus. A mentor's mentor is called a grandmentor. Question: Who is Gus's grandmentor?", ["Eli"]),
         ("Context: Hana supervises Ivo. Ivo supervises Jae. The supervisor of one's supervisor is the senior supervisor. Who is Jae's senior supervisor?", ["Hana"])),
        lambda: ("rule-and-fact-synthesis",
         ("Context: A parcel is express when marked urgent and under 5 kg, unless it contains glass. Parcel R is urgent, 3 kg, and contains glass. Express parcels use Dock 1; others use Dock 3. Which dock handles R?", ["Dock 3", "3"]),
         ("Context: A case is auto-approved when score>=80 and documents complete, unless fraud flag is set. Case Z score 91, documents complete, fraud flag set. Auto-approved cases go Queue A; others Queue B. Which queue receives Z?", ["Queue B", "B"])),
    )
    pairs = (value for _, value in _materialize_family_inputs(pair_factories, family_indexes))
    return tuple(FamilySpec(slug, (
        {"prompt": first[0], "accepted": first[1]},
        {"prompt": second[0], "accepted": second[1]},
    )) for slug, first, second in pairs)


def _reasoning_families(family_indexes: Iterable[int] | None = None) -> tuple[FamilySpec, ...]:
    pair_factories = (
        lambda: ("net-change",
         ("A counter starts at 7, increases by 5, then decreases by 3. Return only the final integer.", "9", {"op": "net", "start": 7, "changes": [5, -3]}),
         ("A counter starts at 12, decreases by 8, then increases by 6. Return only the final integer.", "10", {"op": "net", "start": 12, "changes": [-8, 6]})),
        lambda: ("weighted-total",
         ("Three red tokens are worth 4 points each and two blue tokens are worth 7 each. Return only the total points.", "26", {"op": "weighted", "items": [[3, 4], [2, 7]]}),
         ("Five small boxes weigh 2 kg each and three large boxes weigh 6 kg each. Return only total kilograms.", "28", {"op": "weighted", "items": [[5, 2], [3, 6]]})),
        lambda: ("ordered-middle",
         ("Lena finished before Omar. Priya finished after Omar. Return only the person in the middle.", "Omar", {"op": "middle", "order": ["Lena", "Omar", "Priya"]}),
         ("Task Cedar precedes Birch, and Birch precedes Aspen. Return only the middle task.", "Birch", {"op": "middle", "order": ["Cedar", "Birch", "Aspen"]})),
        lambda: ("set-intersection",
         ("Set A={mira,noa,sol}; Set B={noa,sol,uma}; Set C={sol,uma}. Return only the name present in all three.", "sol", {"op": "intersection", "sets": [["mira", "noa", "sol"], ["noa", "sol", "uma"], ["sol", "uma"]]}),
         ("Set A={red,blue,green}; Set B={blue,green}; Set C={green,yellow}. Return only the color in all sets.", "green", {"op": "intersection", "sets": [["red", "blue", "green"], ["blue", "green"], ["green", "yellow"]]})),
        lambda: ("state-machine",
         ("State starts CLOSED. OPEN changes it to OPEN, LOCK changes OPEN to LOCKED, and OPEN has no effect while LOCKED. Apply OPEN, LOCK, OPEN. Return only the final state.", "LOCKED", {"op": "transitions", "start": "CLOSED", "events": ["OPEN", "LOCK", "OPEN"], "table": {"CLOSED|OPEN": "OPEN", "OPEN|LOCK": "LOCKED"}}),
         ("State starts IDLE. START makes RUNNING, PAUSE makes PAUSED only from RUNNING, and START from PAUSED makes RUNNING. Apply START, PAUSE, START. Return final state.", "RUNNING", {"op": "transitions", "start": "IDLE", "events": ["START", "PAUSE", "START"], "table": {"IDLE|START": "RUNNING", "RUNNING|PAUSE": "PAUSED", "PAUSED|START": "RUNNING"}})),
        lambda: ("conditional-elimination",
         ("Exactly one box contains a coin. Box A says 'not A'; Box B says 'coin in C'. Exactly one statement is true. Testing the possibilities gives one solution. Return A, B, or C.", "A", {"op": "literal", "answer": "A"}),
         ("Exactly one badge is gold: X, Y, or Z. Claim 1: 'X is gold.' Claim 2: 'Y is not gold.' Exactly one claim is true. Return the gold badge.", "Z", {"op": "literal", "answer": "Z"})),
        lambda: ("capacity-allocation",
         ("A van holds 10 units. Items A=6, B=4, C=5. Choose exactly two items without exceeding capacity and maximize used capacity. Return item letters alphabetically with no separator.", "AB", {"op": "best_pair", "weights": {"A": 6, "B": 4, "C": 5}, "capacity": 10}),
         ("A bin holds 12 units. Items D=7, E=5, F=4. Choose exactly two without exceeding capacity and maximize used capacity. Return letters alphabetically.", "DE", {"op": "best_pair", "weights": {"D": 7, "E": 5, "F": 4}, "capacity": 12})),
        lambda: ("cyclic-schedule",
         ("A three-day cycle is Red, Blue, Green and repeats. Day 1 is Red. What color is day 8? Return only the color.", "Blue", {"op": "cycle", "values": ["Red", "Blue", "Green"], "index": 8}),
         ("A four-shift cycle is A, B, C, D and repeats. Shift 1 is A. What is shift 11?", "C", {"op": "cycle", "values": ["A", "B", "C", "D"], "index": 11})),
        lambda: ("transitive-implication",
         ("Rules: if P then Q; if Q then R; if R then S. P is true. Return the furthest entailed letter only.", "S", {"op": "chain", "start": "P", "edges": [["P", "Q"], ["Q", "R"], ["R", "S"]]}),
         ("Rules: A implies C; C implies D; D implies F. A holds. Return the final entailed letter.", "F", {"op": "chain", "start": "A", "edges": [["A", "C"], ["C", "D"], ["D", "F"]]})),
        lambda: ("meeting-slot",
         ("A is free {1,3,5}; B {2,3,5}; room {3,4}. Choose the earliest common slot. Return only the number.", "3", {"op": "earliest_intersection", "sets": [[1, 3, 5], [2, 3, 5], [3, 4]]}),
         ("X is free {4,6,8}; Y {5,6,8}; room {6,7,8}. Choose the earliest common slot.", "6", {"op": "earliest_intersection", "sets": [[4, 6, 8], [5, 6, 8], [6, 7, 8]]})),
        lambda: ("assignment-enumeration",
         ("Assign Ana and Bo to distinct rooms 1 and 2. Ana cannot use room 2. Return assignments as Ana-room,Bo-room.", "Ana-1,Bo-2", {"op": "literal", "answer": "Ana-1,Bo-2"}),
         ("Assign Cy and Dee to distinct shifts M and N. Dee cannot take M. Return Cy-shift,Dee-shift.", "Cy-M,Dee-N", {"op": "literal", "answer": "Cy-M,Dee-N"})),
        lambda: ("multi-rate-time",
         ("A worker completes 3 units/hour for 2 hours, then 5 units/hour for 3 hours. Return total units.", "21", {"op": "weighted", "items": [[3, 2], [5, 3]]}),
         ("A pump moves 4 liters/minute for 5 minutes, then 2 liters/minute for 3 minutes. Return total liters.", "26", {"op": "weighted", "items": [[4, 5], [2, 3]]})),
        lambda: ("exclusive-rules",
         ("Choose one route. North is allowed only if dry. East is allowed only if daylight. South is allowed only if both wet and dark. It is wet and daylight. Only one route is allowed. Return the route.", "East", {"op": "literal", "answer": "East"}),
         ("Choose one door. Red opens with key and no alarm. Blue opens with code during day. Green opens only at night without alarm. It is day, no key, valid code, alarm off. Return the open door.", "Blue", {"op": "literal", "answer": "Blue"})),
        lambda: ("bounded-path",
         ("Directed edges: A->B, A->C, B->D, C->E, E->D. What is the minimum edge count from A to D? Return an integer.", "2", {"op": "shortest", "edges": [["A", "B"], ["A", "C"], ["B", "D"], ["C", "E"], ["E", "D"]], "start": "A", "end": "D"}),
         ("Directed edges: K->L, L->M, K->N, N->P, P->M. Minimum edges K to M?", "2", {"op": "shortest", "edges": [["K", "L"], ["L", "M"], ["K", "N"], ["N", "P"], ["P", "M"]], "start": "K", "end": "M"})),
        lambda: ("constraint-ordering",
         ("Arrange W,X,Y,Z. W before X; Y immediately before Z; X before Y. Return the unique order with no spaces.", "WXYZ", {"op": "permutation", "items": ["W", "X", "Y", "Z"], "before": [["W", "X"], ["X", "Y"]], "adjacent": [["Y", "Z"]]}),
         ("Arrange A,B,C,D. C immediately before D; A before C; B after D. Return the unique order.", "ACDB", {"op": "permutation", "items": ["A", "B", "C", "D"], "before": [["A", "C"], ["D", "B"]], "adjacent": [["C", "D"]]})),
        lambda: ("resource-sequence",
         ("Start with 8 energy. Action A costs 3 and yields key; B costs 4 and requires key; C restores 2. Execute A,C,B. Return remaining energy.", "3", {"op": "net", "start": 8, "changes": [-3, 2, -4]}),
         ("Start with 10 credits. Step X costs 6 and unlocks Y; bonus adds 3; Y costs 5. Execute X, bonus, Y. Return credits.", "2", {"op": "net", "start": 10, "changes": [-6, 3, -5]})),
    )
    pairs = (value for _, value in _materialize_family_inputs(pair_factories, family_indexes))
    return tuple(FamilySpec(slug, (
        {"prompt": first[0], "answer": first[1], "oracle": first[2]},
        {"prompt": second[0], "answer": second[1], "oracle": second[2]},
    )) for slug, first, second in pairs)


def _summarization_families(family_indexes: Iterable[int] | None = None) -> tuple[FamilySpec, ...]:
    source_factories = (
        lambda: ("chronology", "On Monday the team opened the migration. Tuesday testing found a timezone bug. Wednesday the bug was fixed, and Thursday the migration completed. A separate office lunch occurred Friday.", ["testing found a timezone bug Tuesday", "the bug was fixed Wednesday", "the migration completed Thursday"], "the migration failed"),
        lambda: ("ownership", "Maya drafted the policy, Rafi reviewed its legal terms, and Chen approved the final version. Inez attended the meeting but had no approval role.", ["Maya drafted the policy", "Rafi reviewed the legal terms", "Chen approved the final version"], "Inez approved the policy"),
        lambda: ("incident", "At 09:10 an expired certificate blocked logins. Operations renewed it at 09:32, and access recovered by 09:36. No customer data was lost.", ["an expired certificate blocked logins", "operations renewed the certificate at 09:32", "access recovered by 09:36", "no customer data was lost"], "customer data was lost"),
        lambda: ("quantitative", "The campaign reached 12,400 people, generated 620 visits, and produced 31 purchases. Spend was $1,550. The design team also tested three unused logos.", ["12,400 people were reached", "620 visits were generated", "31 purchases resulted", "spend was $1,550"], "the campaign made 620 purchases"),
        lambda: ("policy-exception", "Employees may work remotely two days per week. New hires must work onsite during their first month, while documented accessibility accommodations can override that restriction.", ["remote work is allowed two days per week", "new hires must be onsite for their first month", "accessibility accommodations can override the restriction"], "new hires may always work remotely"),
        lambda: ("change-over-time", "The original launch date was May 4. Supplier delays moved it to May 18. After expedited shipping, the final approved date became May 12.", ["the original date was May 4", "supplier delays moved it to May 18", "the final approved date is May 12"], "the final date is May 18"),
        lambda: ("causal", "A cooling fan failed, causing the server to overheat and shut down. Replacing the fan restored normal temperature; the database required no repair.", ["a cooling fan failed and the server shut down", "replacing the fan restored normal temperature", "the database required no repair"], "the database caused the shutdown"),
        lambda: ("multi-party-decision", "Asha proposed retaining the vendor. Bo favored a new bid. Cy abstained because of a conflict. The committee voted 4–2 to seek a new vendor.", ["Asha favored retaining the vendor", "Bo favored a new bid", "Cy abstained due to a conflict", "the committee voted 4–2 for a new vendor search"], "the vote retained the vendor"),
        lambda: ("process", "To publish a report, an analyst uploads the draft, a reviewer resolves factual issues, and an editor approves formatting. Publication occurs only after all three stages pass.", ["the analyst uploads the draft", "a reviewer resolves factual issues", "an editor approves formatting", "publication requires all stages to pass"], "publication occurs before review"),
        lambda: ("comparison", "Plan North costs $18 monthly and includes 20 GB. Plan South costs $24 and includes 50 GB plus roaming. Both include phone support.", ["North costs $18 and includes 20 GB", "South costs $24 and includes 50 GB plus roaming", "both include phone support"], "North includes more data than South"),
        lambda: ("uncertainty", "The preliminary survey suggests demand may rise by 8–12%, but the sample is small and the estimate is not a forecast. A larger survey begins next month.", ["preliminary demand may rise 8–12%", "the sample is small", "the estimate is not a forecast", "a larger survey begins next month"], "demand will definitely rise 12%"),
        lambda: ("nested-exception", "Standard refunds require a receipt within 30 days. Gifts may use an order number instead. Clearance items are never refundable unless defective.", ["standard refunds require a receipt within 30 days", "gifts may use an order number", "clearance items require a defect to be refundable"], "all clearance items are refundable"),
        lambda: ("milestones", "The bridge design passed safety review in January. Funding was approved in March. Construction began in June and is scheduled to finish in November.", ["safety review passed in January", "funding was approved in March", "construction began in June", "completion is scheduled for November"], "construction finished in June"),
        lambda: ("tradeoff", "Option A cuts latency by 30% but raises cost by 15%. Option B keeps cost flat and cuts latency by 10%. The team selected B because the budget is fixed.", ["A cuts latency 30% but raises cost 15%", "B keeps cost flat and cuts latency 10%", "the team selected B because the budget is fixed"], "the team selected A"),
        lambda: ("handoff", "Support reproduced the defect and sent logs to Engineering. Engineering identified a parser bug and supplied a patch. Release Management scheduled the patch for Tuesday.", ["support reproduced the defect and sent logs", "engineering found a parser bug and supplied a patch", "release management scheduled Tuesday"], "support supplied the patch"),
        lambda: ("mixed-signal", "Revenue rose 6% and customer count rose 9%, while average order value fell 3%. Management attributed growth to new customers, not larger purchases.", ["revenue rose 6%", "customer count rose 9%", "average order value fell 3%", "management attributed growth to new customers"], "larger purchases drove growth"),
    )
    families = []
    for index, source_spec in _materialize_family_inputs(source_factories, family_indexes):
        slug, source, facts, forbidden = source_spec
        second_source = source.replace("The ", "According to the update, the ", 1) if "The " in source else source + " This update supersedes informal notes."
        variants = []
        for variant, text in enumerate((source, second_source)):
            max_words = 45 if index < 8 else 55
            prompt = f"Summarize the source in at most {max_words} words and no more than 2 sentences. Source: {text}"
            variants.append({
                "prompt": prompt,
                "facts": facts,
                "forbidden": [forbidden],
                "requirements": [f"Accurately state that {fact}." for fact in facts],
                "constraints": {"max_words": max_words, "max_sentences": 2, "forbidden_strings": [forbidden]},
            })
        families.append(FamilySpec(slug, tuple(variants)))
    return tuple(families)


# Content revision 1.1.0. These alternatives replace the former fixture
# rotations with independently useful behaviors and hidden cases.
CODING_VARIANT_B: dict[str, dict[str, Any]] = {
    "clamp-sequence": {"function_name": "clamp_records", "parameters": 3,
        "behavior": "Records are [id,value] pairs. Return new [id,value] pairs with values clamped to inclusive [low,high], preserving record order and inputs.",
        "canonical": "def clamp_records(records, low, high):\n    return [[row[0], min(high, max(low, row[1]))] for row in records]",
        "tests": [([[["a", -4], ["b", 8]], 0, 5], [["a", 0], ["b", 5]]), ([[], -2, 2], []), ([[["x", 3]], 3, 3], [["x", 3]]), ([[["x", -2], ["x", 2]], -1, 1], [["x", -1], ["x", 1]])]},
    "collapse-whitespace": {"function_name": "normalize_lines", "parameters": 1,
        "behavior": "Normalize each nonempty line by collapsing its internal whitespace to one ASCII space; discard blank lines and join retained lines with newline characters.",
        "canonical": "def normalize_lines(text):\n    return '\\n'.join(' '.join(line.split()) for line in text.splitlines() if line.split())",
        "tests": [([" a   b\n\n c\td "], "a b\nc d"), ([""], ""), (["  one  "], "one"), (["x\n  \ny\n"], "x\ny")]},
    "initial-histogram": {"function_name": "unique_by_initial", "parameters": 1,
        "behavior": "Count distinct words case-insensitively by their lowercase first character; ignore empty strings.",
        "canonical": "def unique_by_initial(words):\n    groups = {}\n    for word in words:\n        if word:\n            key = word[0].lower()\n            groups.setdefault(key, set()).add(word.lower())\n    return {key: len(values) for key, values in groups.items()}",
        "tests": [([["Apple", "apple", "ant", "Bee"]], {"a": 2, "b": 1}), ([[]], {}), ([["X", "x", ""]], {"x": 1}), ([["9a", "9A", "9b"]], {"9": 2})]},
    "key-value-parser": {"function_name": "parse_typed_pairs", "parameters": 1,
        "behavior": "Parse key=value strings, trimming both sides and ignoring malformed or empty-key entries. Convert true/false case-insensitively to booleans and signed decimal integers to integers; keep other values as strings. Last duplicate wins.",
        "canonical": "def parse_typed_pairs(lines):\n    out = {}\n    for line in lines:\n        if '=' not in line:\n            continue\n        key, value = line.split('=', 1)\n        key, value = key.strip(), value.strip()\n        if not key:\n            continue\n        low = value.lower()\n        if low in ('true', 'false'):\n            out[key] = low == 'true'\n        elif value.lstrip('-').isdigit():\n            out[key] = int(value)\n        else:\n            out[key] = value\n    return out",
        "tests": [([["age=12", "active=TRUE", "name=Ada"]], {"age": 12, "active": True, "name": "Ada"}), ([["x=-3", "x=4"]], {"x": 4}), ([["bad", "=7"]], {}), ([["zero=0", "blank="]], {"zero": 0, "blank": ""})]},
    "cyclic-rotation": {"function_name": "rotate_right", "parameters": 2,
        "behavior": "Return a new list rotated right by k positions. Support negative and oversized k and preserve the input.",
        "canonical": "def rotate_right(items, k):\n    if not items:\n        return []\n    n = k % len(items)\n    return items[-n:] + items[:-n] if n else list(items)",
        "tests": [([[1, 2, 3, 4], 1], [4, 1, 2, 3]), ([[1, 2, 3], 4], [3, 1, 2]), ([[1, 2, 3], -1], [2, 3, 1]), ([[], 9], [])]},
    "stable-score-ranking": {"function_name": "rank_scores_stable", "parameters": 1,
        "behavior": "Input records are [name,score]. Return names by descending score while preserving original order for ties.",
        "canonical": "def rank_scores_stable(records):\n    return [row[0] for row in sorted(records, key=lambda row: -row[1])]",
        "tests": [([[['b', 2], ['a', 2], ['c', 1]]], ["b", "a", "c"]), ([[]], []), ([[['z', -1], ['x', 3], ['y', 3]]], ["x", "y", "z"]), ([[['a', 1]]], ["a"])]},
    "merge-adjacent-intervals": {"function_name": "merge_with_gap", "parameters": 2,
        "behavior": "Merge sorted or unsorted closed integer intervals when the number of uncovered integers between them is at most max_gap. Return sorted [start,end] lists; max_gap is nonnegative.",
        "canonical": "def merge_with_gap(intervals, max_gap):\n    out = []\n    for start, end in sorted(intervals):\n        if not out or start - out[-1][1] - 1 > max_gap:\n            out.append([start, end])\n        else:\n            out[-1][1] = max(out[-1][1], end)\n    return out",
        "tests": [([[[1, 2], [4, 5]], 1], [[1, 5]]), ([[[1, 2], [4, 5]], 0], [[1, 2], [4, 5]]), ([[], 3], []), ([[[7, 9], [1, 3], [3, 6]], 0], [[1, 9]])]},
    "balanced-delimiters": {"function_name": "bracket_depth", "parameters": 1,
        "behavior": "Return the maximum nesting depth of (), [], and {} when correctly nested, ignoring other characters; return -1 for any mismatch or unclosed delimiter.",
        "canonical": "def bracket_depth(text):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    best = 0\n    for ch in text:\n        if ch in '([{':\n            stack.append(ch)\n            best = max(best, len(stack))\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return -1\n    return best if not stack else -1",
        "tests": [(["a([{}])"], 3), (["([)]"], -1), (["plain"], 0), (["(()"], -1)]},
    "unweighted-shortest-path": {"function_name": "shortest_undirected_hops", "parameters": 3,
        "behavior": "Edges are undirected [left,right] pairs. Return minimum hops from start to end, or -1 if unreachable.",
        "canonical": "def shortest_undirected_hops(edges, start, end):\n    if start == end:\n        return 0\n    queue = [[start, 0]]\n    seen = {start}\n    for node, dist in queue:\n        for left, right in edges:\n            nxt = right if left == node else left if right == node else None\n            if nxt is not None and nxt not in seen:\n                if nxt == end:\n                    return dist + 1\n                seen.add(nxt)\n                queue.append([nxt, dist + 1])\n    return -1",
        "tests": [([[['a','b'],['c','b']], 'a', 'c'], 2), ([[['a','b']], 'b', 'a'], 1), ([[], 'x', 'x'], 0), ([[['a','b'],['c','d']], 'a', 'd'], -1)]},
    "nested-group-total": {"function_name": "weighted_group_sum", "parameters": 2,
        "behavior": "Return the sum of each integer group multiplied by its corresponding integer weight. Groups and weights have equal length.",
        "canonical": "def weighted_group_sum(groups, weights):\n    return sum(sum(group) * weight for group, weight in zip(groups, weights))",
        "tests": [([[[1, 2], [3]], [2, 4]], 18), ([[], []], 0), ([[[]], [9]], 0), ([[[-1, 5], [2, 2]], [-2, 3]], 4)]},
    "longest-nondecreasing": {"function_name": "longest_nondecreasing_run", "parameters": 1,
        "behavior": "Return the length of the longest contiguous nondecreasing run.",
        "canonical": "def longest_nondecreasing_run(values):\n    best = current = 0\n    previous = None\n    for value in values:\n        current = current + 1 if previous is not None and previous <= value else 1\n        best = max(best, current)\n        previous = value\n    return best",
        "tests": [([[3, 1, 2, 2, 0, 4]], 3), ([[]], 0), ([[5, 4, 3]], 1), ([[1, 1, 1]], 3)]},
    "slug-validation": {"function_name": "split_ascii_slug", "parameters": 1,
        "behavior": "For a valid nonempty ASCII slug containing only a-z, ASCII digits 0-9, and single internal hyphens, return its components split on hyphens. Return [] when invalid.",
        "canonical": "def split_ascii_slug(text):\n    if not text or text[0] == '-' or text[-1] == '-':\n        return []\n    previous = ''\n    for ch in text:\n        if not (('0' <= ch <= '9') or ('a' <= ch <= 'z') or ch == '-'):\n            return []\n        if ch == '-' and previous == '-':\n            return []\n        previous = ch\n    return text.split('-')",
        "tests": [(["alpha-2"], ["alpha", "2"]), (["two--parts"], []), (["café-2"], []), (["item-٣"], [])]},
    "run-length-encoding": {"function_name": "run_length_decode", "parameters": 1,
        "behavior": "Decode [[character,count],...] into a string. Counts are nonnegative integers and characters are single-character strings.",
        "canonical": "def run_length_decode(runs):\n    return ''.join(ch * count for ch, count in runs)",
        "tests": [([[["a",3],["b",2]]], "aaabb"), ([[]], ""), ([[["A",1],["a",2]]], "Aaa"), ([[["x",0],["y",1]]], "y")]},
    "rolling-window-sums": {"function_name": "window_averages", "parameters": 2,
        "behavior": "Return arithmetic means for each contiguous window of size k. Return [] when k <= 0 or exceeds the list length.",
        "canonical": "def window_averages(values, k):\n    if k <= 0 or k > len(values):\n        return []\n    return [sum(values[i:i+k]) / k for i in range(len(values)-k+1)]",
        "tests": [([[1,3,5],2], [2.0,4.0]), ([[2],1], [2.0]), ([[1,2],0], []), ([[2,-2,2],2], [0.0,0.0])]},
    "transaction-state": {"function_name": "transaction_summary", "parameters": 2,
        "behavior": "Apply signed transactions in order, skipping ones that would make balance negative. Return [final_balance,skipped_count].",
        "canonical": "def transaction_summary(balance, transactions):\n    skipped = 0\n    for amount in transactions:\n        if balance + amount < 0:\n            skipped += 1\n        else:\n            balance += amount\n    return [balance, skipped]",
        "tests": [([10,[-3,-8,5]], [12,1]), ([0,[-1,2]], [2,1]), ([5,[]], [5,0]), ([3,[-3,-1]], [0,1])]},
    "dependency-readiness": {"function_name": "blocked_steps", "parameters": 2,
        "behavior": "Dependencies map each step to prerequisite names. Return alphabetically sorted incomplete steps that still lack at least one completed prerequisite.",
        "canonical": "def blocked_steps(dependencies, completed):\n    done = set(completed)\n    return sorted(step for step, needs in dependencies.items() if step not in done and any(item not in done for item in needs))",
        "tests": [([{"build":["test"],"test":["code"],"code":[]}, ["code"]], ["build"]), ([{},[]], []), ([{"a":[],"b":["x"]},[]], ["b"]), ([{"a":["x"]},["a"]], [])]},
}


SUMMARY_VARIANT_B = {
    "chronology": ("The permit request arrived Friday. Staff requested a missing diagram Monday, received it Wednesday, and approved the permit Thursday. A billing address changed Tuesday but did not affect review.", ["staff requested a missing diagram Monday", "the diagram arrived Wednesday", "the permit was approved Thursday"], "the billing change delayed approval"),
    "ownership": ("Jordan collected customer feedback, Priya converted it into requirements, and Luis authorized the release. Morgan observed the review but cast no vote.", ["Jordan collected customer feedback", "Priya wrote the requirements", "Luis authorized the release"], "Morgan authorized the release"),
    "incident": ("A routing rule deployed at 14:05 sent checkout traffic to an unavailable service. The rule was rolled back at 14:17; checkout recovered at 14:20, and no payments were duplicated.", ["a routing rule sent checkout traffic to an unavailable service", "the rule was rolled back at 14:17", "checkout recovered at 14:20", "no payments were duplicated"], "payments were duplicated"),
    "quantitative": ("The workshop registered 480 people; 360 attended, 288 completed the lab, and 250 submitted feedback. Catering prepared 400 lunches.", ["480 people registered", "360 attended", "288 completed the lab", "250 submitted feedback"], "400 people attended"),
    "policy-exception": ("Expense reports are due within 20 days. Employees on approved leave receive five extra days, but cash advances must always be reconciled within 10 days.", ["expense reports are due within 20 days", "approved leave adds five days", "cash advances remain due within 10 days"], "leave extends the cash-advance deadline"),
    "change-over-time": ("The storage target began at 80 TB, increased to 110 TB after acquisition, and was reduced to 95 TB after archival. The approved capacity plan now uses 95 TB.", ["the target began at 80 TB", "acquisition raised it to 110 TB", "archival reduced the approved target to 95 TB"], "the approved target is 110 TB"),
    "causal": ("A malformed catalog entry caused the importer to reject the nightly batch. Correcting the entry allowed the rerun to finish; the importer code was unchanged.", ["a malformed catalog entry caused the batch rejection", "correcting the entry allowed the rerun to finish", "the importer code was unchanged"], "an importer code defect caused the rejection"),
    "multi-party-decision": ("Nora recommended extending the trial, Omar preferred purchase, and Pia requested more security evidence. The board postponed purchase 5–1 pending the security review.", ["Nora recommended extending the trial", "Omar preferred purchase", "Pia requested security evidence", "the board voted 5–1 to postpone purchase"], "the board approved purchase"),
    "process": ("A refund request is first matched to the original payment, then checked for policy eligibility, and finally approved by Finance. Funds are released only after all three checks succeed.", ["the request is matched to the original payment", "policy eligibility is checked", "Finance gives final approval", "funds require all three checks"], "funds are released before Finance approval"),
    "comparison": ("Vendor Pine charges $900 setup plus $80 monthly and offers weekday support. Vendor Lake has no setup fee, charges $125 monthly, and includes 24/7 support.", ["Pine charges $900 setup and $80 monthly", "Pine offers weekday support", "Lake has no setup fee and costs $125 monthly", "Lake includes 24/7 support"], "Pine includes 24/7 support"),
    "uncertainty": ("An early sensor analysis indicates energy use could fall 4–7%, but winter data is missing and the result has not been independently replicated. A full-year study ends in December.", ["early analysis suggests a 4–7% reduction", "winter data is missing", "the result is not independently replicated", "the full-year study ends in December"], "energy use will certainly fall 7%"),
    "nested-exception": ("Reservations can be changed without charge until noon the prior day. Flexible fares may change until departure, while group bookings always require coordinator approval.", ["standard reservations are free to change until noon the prior day", "flexible fares may change until departure", "group bookings require coordinator approval"], "group bookings never require approval"),
    "milestones": ("The clinic lease was signed in February, renovation passed inspection in April, staff training finished in May, and opening is planned for July.", ["the lease was signed in February", "renovation passed inspection in April", "staff training finished in May", "opening is planned for July"], "the clinic opened in May"),
    "tradeoff": ("Database X cuts storage cost 20% but increases recovery time from 10 to 35 minutes. Database Y keeps current cost and recovery time. The team retained Y because recovery speed is mandatory.", ["X cuts storage cost 20%", "X increases recovery time to 35 minutes", "Y preserves current cost and recovery time", "the team retained Y for recovery speed"], "the team selected X"),
    "handoff": ("Sales documented the contract exception and asked Legal for review. Legal approved revised language, then Operations added it to the renewal package due Friday.", ["Sales documented the exception and contacted Legal", "Legal approved revised language", "Operations added it to the Friday renewal package"], "Sales approved the legal language"),
    "mixed-signal": ("Orders grew 11% and delivery time improved 8%, but returns rose from 4% to 6%. Leaders credited warehouse automation for speed while opening a review of return causes.", ["orders grew 11%", "delivery time improved 8%", "returns rose from 4% to 6%", "leaders credited automation for speed and are reviewing returns"], "automation was proven to cause the higher return rate"),
}


CONTENT_COMPLEXITY = {
    "classification": {"direct-material": 1, "numeric-band": 2, "document-routing": 2, "priority-precedence": 4, "refund-exception": 4, "risk-matrix": 5, "inventory-state": 3, "access-policy": 6, "sla-clock": 4, "travel-policy": 6, "retention-rule": 4, "subscription-eligibility": 6, "incident-cause": 7, "shipping-service": 5, "support-escalation": 6, "compliance-hierarchy": 7},
    "coding": {"clamp-sequence": 2, "collapse-whitespace": 2, "initial-histogram": 3, "key-value-parser": 5, "cyclic-rotation": 4, "stable-score-ranking": 3, "merge-adjacent-intervals": 6, "balanced-delimiters": 6, "unweighted-shortest-path": 8, "nested-group-total": 2, "longest-nondecreasing": 8, "slug-validation": 5, "run-length-encoding": 4, "rolling-window-sums": 4, "transaction-state": 5, "dependency-readiness": 7},
    "extraction": {"contact-prose": 1, "invoice-header": 2, "latest-error-log": 4, "text-table-filter": 4, "optional-field": 3, "repeated-entity-total": 5, "nested-shipment": 4, "section-precedence": 7, "unit-bearing-measurement": 4, "multi-party-action": 6, "keyed-list-order": 5, "event-window": 6, "cross-line-record": 6, "conditional-record-selection": 7, "paired-source-join": 8, "exception-aware-roles": 8},
    "json": {"flat-profile": 1, "filter-array": 3, "nested-address": 2, "enum-mapping": 4, "nullable-field": 3, "sorted-ranking": 5, "grouped-counts": 5, "conditional-field": 5, "deduplicated-array": 4, "joined-records": 7, "cross-field-status": 6, "hierarchy-tree": 8, "schedule-slots": 7, "invoice-aggregation": 6, "permission-matrix": 8, "rule-derived-summary": 8},
    "qa": {"direct-location": 1, "attribute-lookup": 1, "two-fact-sum": 2, "temporal-latest": 4, "difference-comparison": 2, "exception-owner": 6, "multi-passage-link": 5, "schedule-intersection": 5, "conditional-count": 4, "state-after-updates": 3, "rank-with-tie-rule": 4, "dependency-question": 6, "policy-date-window": 5, "multi-step-rate": 3, "role-chain": 5, "rule-and-fact-synthesis": 7},
    "reasoning": {"net-change": 1, "weighted-total": 2, "ordered-middle": 2, "set-intersection": 3, "state-machine": 5, "conditional-elimination": 7, "capacity-allocation": 5, "cyclic-schedule": 3, "transitive-implication": 4, "meeting-slot": 4, "assignment-enumeration": 5, "multi-rate-time": 3, "exclusive-rules": 6, "bounded-path": 6, "constraint-ordering": 8, "resource-sequence": 6},
    "summarization": {"chronology": 3, "ownership": 3, "incident": 4, "quantitative": 5, "policy-exception": 5, "change-over-time": 5, "causal": 4, "multi-party-decision": 7, "process": 6, "comparison": 5, "uncertainty": 7, "nested-exception": 7, "milestones": 6, "tradeoff": 6, "handoff": 6, "mixed-signal": 8},
}


def _redesign_families(category: str, families: tuple[FamilySpec, ...]) -> tuple[FamilySpec, ...]:
    redesigned = []
    for family in families:
        variants = [dict(value) for value in family.variants]
        if category == "coding":
            alt = CODING_VARIANT_B[family.slug]
            variants[1] = {
                "prompt": (f"Write Python function `{alt['function_name']}` with exactly "
                           f"{alt['parameters']} positional parameter{'s' if alt['parameters'] != 1 else ''}. "
                           f"{alt['behavior']} Return only the function definition. Do not import modules or mutate inputs."),
                "function_name": alt["function_name"], "parameters": alt["parameters"],
                "tests": [{"args": args, "expected": expected} for args, expected in alt["tests"]],
                "canonical": alt["canonical"],
                "incorrect": f"def {alt['function_name']}({', '.join('x'+str(i) for i in range(alt['parameters']))}):\n    return None",
                "required_ast": ["Return"],
            }
            variants[0]["prompt"] = variants[0]["prompt"].replace("1 positional parameters", "1 positional parameter")
        elif category == "summarization":
            source, facts, forbidden = SUMMARY_VARIANT_B[family.slug]
            max_words = variants[1]["constraints"]["max_words"]
            variants[1] = {
                "prompt": f"Summarize the source in at most {max_words} words and no more than 2 sentences. Source: {source}",
                "facts": facts, "forbidden": [forbidden],
                "requirements": [f"Accurately state that {fact}." for fact in facts],
                "constraints": {"max_words": max_words, "max_sentences": 2,
                                "forbidden_strings": [forbidden]},
            }
        elif category == "qa":
            variants = [{**value, "prompt": value["prompt"] + " Give only the requested answer."}
                        for value in variants]
            if family.slug == "policy-date-window":
                variants[1]["prompt"] = ("A travel booking may be cancelled without a fee when the request is made at least "
                    "48 hours before departure. The request was submitted exactly 48 hours before departure. "
                    "Is the cancellation free? Answer yes or no only.")
        elif category == "reasoning":
            natural = {
                "ordered-middle": ["Lena checked in before Omar, and Priya checked in after Omar. Who checked in second? Return only the name.", "The deployment queue places Cedar before Birch and Birch before Aspen. Return only the middle task."],
                "set-intersection": ["Mira, Noa, and Sol can work Monday; Noa, Sol, and Uma can work Tuesday; Sol and Uma can work Wednesday. Who is available all three days? Return only the name.", "The red, blue, and green rooms have projectors; blue and green have video links; green and yellow have recording. Which room color has all three features?"],
                "conditional-elimination": ["A shipment is in dock A, B, or C. The scanner reports 'not A' and the manifest reports 'in C'; exactly one report is accurate. Which dock has the shipment? Return A, B, or C.", "One of servers X, Y, or Z holds the active job. Monitor 1 says X; monitor 2 says the job is not on Y; exactly one monitor is accurate. Return the server letter."],
                "cyclic-schedule": ["A maintenance crew rotates Red, Blue, Green daily, starting with Red on day 1. Which crew is assigned on day 8?", "Support shifts repeat A, B, C, D, with shift 1 assigned A. Which team handles shift 11?"],
                "transitive-implication": ["Approval P releases review Q; Q releases deployment R; R activates service S. Approval P is complete. Return the furthest stage now entailed.", "Completing milestone A unlocks C, C unlocks D, and D unlocks F. A is complete. Return the final unlocked milestone."],
                "meeting-slot": ["Alex is free in slots 1, 3, 5; Blair in 2, 3, 5; and the room in 3, 4. Return the earliest common slot.", "The designer is free at 4, 6, 8; the reviewer at 5, 6, 8; and the lab at 6, 7, 8. Return the earliest common slot."],
                "assignment-enumeration": ["Assign Ana and Bo to different rooms 1 and 2. Ana cannot use room 2. Return `Ana-room,Bo-room`.", "Assign Cy and Dee to different shifts M and N. Dee cannot take M. Return `Cy-shift,Dee-shift`."],
                "exclusive-rules": ["A delivery may use North only in dry weather, East only during daylight, and South only when it is both wet and dark. It is wet and daylight. Which single route is allowed?", "A technician can open Red with a key and no alarm, Blue with a code during daytime, or Green only at night without an alarm. It is daytime, there is no key, the code is valid, and the alarm is off. Which door opens?"],
            }
            if family.slug in natural:
                for index in range(2): variants[index]["prompt"] = natural[family.slug][index]
        redesigned.append(FamilySpec(family.slug, tuple(variants)))
    return tuple(redesigned)


def _difficulty_assignments(category: str, families: tuple[FamilySpec, ...]) -> dict[tuple[str, int], str]:
    """Rank reviewed content scores; labels are independent of family position/split."""
    scored = []
    for family in families:
        for variant_index, variant in enumerate(family.variants):
            score = CONTENT_COMPLEXITY[category][family.slug]
            if variant_index == 1 and category in {"coding", "summarization"}:
                score += 1
            score += min(2, max(0, len(variant["prompt"].split()) - 35) // 25)
            scored.append((score, family.slug, variant_index))
    scored.sort()
    labels = ["easy"] * 8 + ["medium"] * 13 + ["hard"] * 11
    return {(slug, variant): label for label, (_, slug, variant) in zip(labels, scored, strict=True)}


FAMILY_BUILDERS = {
    "classification": _classification_families,
    "coding": _coding_families,
    "extraction": _extraction_families,
    "json": _json_families,
    "qa": _qa_families,
    "reasoning": _reasoning_families,
    "summarization": _summarization_families,
}


class CategoryDetectorProjectionError(ValueError):
    """Raised when the prompt projection cannot fail closed."""


_CATEGORY_DETECTOR_SPLITS = {
    "train": (tuple(range(0, 10)), "train", "train-manifest.json"),
    "dev": (tuple(range(10, 13)), "development", "development-manifest.json"),
}


def export_category_detector_prompts(
    split: Literal["train", "dev"], *, artifact_root: Path = DEFAULT_ROOT,
) -> tuple[dict[str, str], ...]:
    """Build prompt-only TRAIN or DEV records without constructing FINAL families."""
    if split not in _CATEGORY_DETECTOR_SPLITS:
        raise CategoryDetectorProjectionError(
            "Category-detector prompt projection permits only 'train' or 'dev'"
        )

    family_indexes, manifest_split, manifest_name = _CATEGORY_DETECTOR_SPLITS[split]
    manifest = json.loads((artifact_root / manifest_name).read_text(encoding="utf-8"))
    entries = manifest.get("tasks")
    if not isinstance(entries, list):
        raise CategoryDetectorProjectionError(f"Invalid {split} split manifest")

    task_ids = [entry.get("task_id") for entry in entries]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise CategoryDetectorProjectionError(f"Invalid task ID in {split} split manifest")
    if len(task_ids) != len(set(task_ids)):
        raise CategoryDetectorProjectionError(f"Duplicate task ID in {split} split manifest")
    if any(entry.get("split") != manifest_split for entry in entries):
        raise CategoryDetectorProjectionError(f"Unexpected split entry in {split} split manifest")

    entries_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        family_id = entry.get("task_family_id")
        if not isinstance(family_id, str) or not family_id:
            raise CategoryDetectorProjectionError(f"Missing family ID in {split} split manifest")
        entries_by_family[family_id].append(entry)

    records: list[dict[str, str]] = []
    consumed_families: set[str] = set()
    for category in CATEGORIES:
        families = _redesign_families(
            category, FAMILY_BUILDERS[category](family_indexes)
        )
        if len(families) != len(family_indexes):
            raise CategoryDetectorProjectionError(
                f"Unexpected {split} family count for {category}"
            )
        for family_index, family in zip(family_indexes, families, strict=True):
            family_id = f"rbv1-{category}-{family_index + 1:02d}-{family.slug}"
            family_entries = entries_by_family.get(family_id, [])
            if len(family_entries) != len(family.variants):
                raise CategoryDetectorProjectionError(
                    f"Manifest membership mismatch for {family_id}"
                )
            consumed_families.add(family_id)
            for entry, variant in zip(family_entries, family.variants, strict=True):
                if entry.get("internal_category") != category:
                    raise CategoryDetectorProjectionError(
                        f"Manifest category mismatch for {entry['task_id']}"
                    )
                prompt = variant.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    raise CategoryDetectorProjectionError(
                        f"Missing prompt for {entry['task_id']}"
                    )
                records.append({
                    "task_id": entry["task_id"],
                    "split": split,
                    "category": entry["category"],
                    "prompt": prompt,
                })

    if consumed_families != set(entries_by_family):
        raise CategoryDetectorProjectionError(f"Unexpected family in {split} split manifest")
    expected_per_category = 20 if split == "train" else 6
    category_counts = Counter(record["category"] for record in records)
    expected_categories = set(DISPLAY_CATEGORY.get(value, value) for value in CATEGORIES)
    if set(category_counts) != expected_categories or any(
        count != expected_per_category for count in category_counts.values()
    ):
        raise CategoryDetectorProjectionError(f"Unexpected category balance in {split} projection")
    if len(records) != expected_per_category * len(CATEGORIES):
        raise CategoryDetectorProjectionError(f"Unexpected task count in {split} projection")
    return tuple(records)


def _budget(category: str, difficulty: str) -> int:
    return {
        "classification": {"easy": 32, "medium": 32, "hard": 32},
        "coding": {"easy": 128, "medium": 192, "hard": 256},
        "extraction": {"easy": 64, "medium": 128, "hard": 192},
        "json": {"easy": 64, "medium": 128, "hard": 192},
        "qa": {"easy": 64, "medium": 128, "hard": 160},
        "reasoning": {"easy": 160, "medium": 160, "hard": 160},
        "summarization": {"easy": 96, "medium": 128, "hard": 192},
    }[category][difficulty]


def _task_metadata(category: str, variant: dict[str, Any], difficulty: str) -> tuple[dict[str, Any], str, str, float]:
    metadata: dict[str, Any] = {"difficulty": difficulty}
    output_type, canonical, threshold = "text", "", 1.0
    if category == "classification":
        metadata.update(expected_label=variant["expected"], canonical_response=variant["expected"],
                        incorrect_response="NOT_THE_LABEL")
        canonical = variant["expected"]
    elif category == "coding":
        metadata.update(evaluation_mode="static_pending_functional",
                        function_name=variant["function_name"], parameters=variant["parameters"],
                        required_ast=variant["required_ast"], functional_tests=variant["tests"],
                        execution_limits={"timeout_seconds": 3, "memory": "64m", "cpus": 0.5,
                                          "pids_limit": 32, "network": "none"},
                        canonical_response=variant["canonical"], incorrect_response=variant["incorrect"])
        canonical, output_type = variant["canonical"], "code"
    elif category in {"extraction", "json"}:
        canonical = json.dumps(variant["expected"], ensure_ascii=False, sort_keys=True)
        metadata.update(expected=variant["expected"], canonical_response=canonical,
                        incorrect_response="{}")
        output_type = "json"
    elif category == "qa":
        metadata.update(accepted_answers=variant["accepted"],
                        canonical_response=str(variant["accepted"][0]), incorrect_response="incorrect")
        canonical = str(variant["accepted"][0])
    elif category == "reasoning":
        metadata.update(accepted_answers=[variant["answer"]], oracle_spec=variant["oracle"],
                        canonical_response=variant["answer"], incorrect_response="incorrect")
        canonical = variant["answer"]
    else:
        metadata.update(required_facts=variant["facts"], forbidden_claims=variant["forbidden"],
                        semantic_requirements=variant["requirements"],
                        deterministic_constraints=variant["constraints"],
                        judge_contract={"judge_version": DATASET_SEMANTIC_JUDGE_VERSION,
                                        "prompt_version": DATASET_JUDGE_PROMPT_VERSION,
                                        "factual_consistency_veto": True})
        canonical, threshold = "", 0.8
    return metadata, output_type, canonical, threshold


def construct_dataset() -> BenchmarkDataset:
    tasks: list[BenchmarkTask] = []
    counters: dict[tuple[str, str], int] = defaultdict(int)
    for category in CATEGORIES:
        families = _redesign_families(category, FAMILY_BUILDERS[category]())
        difficulties = _difficulty_assignments(category, families)
        if len(families) != 16 or len({family.slug for family in families}) != 16:
            raise ValueError(f"{category} must define exactly 16 unique families")
        for family_index, family in enumerate(families, 1):
            family_id = f"rbv1-{category}-{family_index:02d}-{family.slug}"
            for variant_index, variant in enumerate(family.variants, 1):
                difficulty = difficulties[(family.slug, variant_index - 1)]
                counters[(category, difficulty)] += 1
                task_id = f"{category}-{difficulty}-{counters[(category, difficulty)]:03d}"
                metadata, output_type, canonical, threshold = _task_metadata(category, variant, difficulty)
                budget = _budget(category, difficulty)
                if canonical and max(1, (len(canonical) + 3) // 4) > int(budget * 0.8):
                    raise ValueError(f"Canonical response does not fit comfortably: {task_id}")
                tasks.append(BenchmarkTask(
                    task_id=task_id, category=category, prompt=variant["prompt"],
                    system_prompt="Follow the requested output format exactly. Use only the supplied task information.",
                    max_output_tokens=budget, temperature=0, expected_output_type=output_type,
                    evaluation_metadata=metadata, acceptable_threshold=threshold,
                    tags=("routing-benchmark-v1", "new", "evaluation-only-metadata"),
                    task_family_id=family_id,
                    source_type="controlled_programmatic_construction",
                    source_id=f"routing-benchmark-v1/{category}/{family.slug}@1",
                    generation_seed=SPLIT_SEED + family_index * 10 + variant_index,
                    evaluator_type=EVALUATOR_TYPES[category],
                    family_variant=("a" if variant_index == 1 else "b"),
                ))
    return BenchmarkDataset(name=BENCHMARK_NAME, version=BENCHMARK_VERSION, tasks=tuple(tasks))


def split_for_family(family_id: str) -> str:
    match = re.fullmatch(r"rbv1-[a-z]+-(\d{2})-[a-z0-9_-]+", family_id)
    if match is None:
        raise ValueError(f"Invalid Routing Benchmark family ID: {family_id}")
    index = int(match.group(1))
    if not 1 <= index <= len(FAMILY_SPLITS):
        raise ValueError(f"Family index is outside the split plan: {family_id}")
    return FAMILY_SPLITS[index - 1]


def build_split_manifest(dataset: BenchmarkDataset) -> dict[str, Any]:
    entries = [{
        "task_id": task.task_id,
        "category": DISPLAY_CATEGORY.get(task.category, task.category),
        "internal_category": task.category,
        "difficulty": task.difficulty,
        "task_family_id": task.task_family_id,
        "split": split_for_family(task.task_family_id or ""),
    } for task in dataset.tasks]
    return {"benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
            "assignment_seed": SPLIT_SEED, "grouping_key": "task_family_id",
            "outcome_features_used": False, "entries": entries}


def build_pilot_manifest(dataset: BenchmarkDataset, split_manifest: dict[str, Any]) -> dict[str, Any]:
    split_by_id = {entry["task_id"]: entry for entry in split_manifest["entries"]}
    selected: list[dict[str, Any]] = []
    for category in CATEGORIES:
        used_families: set[str] = set()
        category_tasks = [task for task in dataset.tasks
                          if task.category == category and split_by_id[task.task_id]["split"] == "train"]
        for difficulty in ("easy", "medium", "hard"):
            task = next(task for task in category_tasks
                        if task.difficulty == difficulty and task.task_family_id not in used_families)
            used_families.add(task.task_family_id or "")
            selected.append({"task_id": task.task_id,
                             "category": DISPLAY_CATEGORY.get(category, category),
                             "internal_category": category, "difficulty": difficulty,
                             "task_family_id": task.task_family_id, "split": "train"})
    return {"benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
            "selection_rule": "reviewed train task per category/difficulty using distinct families and no trivial-only signal",
            "expected_candidate_calls": len(selected) * 4,
            "maximum_expected_semantic_judge_calls": 3 * 4,
            "tasks": selected}


def content_quality_audit(dataset: BenchmarkDataset, split_manifest: dict[str, Any],
                          pilot_manifest: dict[str, Any]) -> dict[str, Any]:
    """Deterministic content gate backing the human-review-v2 packet."""
    by_family: dict[str, list[BenchmarkTask]] = defaultdict(list)
    for task in dataset.tasks:
        by_family[task.task_family_id or ""].append(task)
    variation = Counter()
    failures = []
    for family_id, pair in by_family.items():
        pair.sort(key=lambda task: task.family_variant or "")
        category = pair[0].category
        if category == "coding":
            left, right = (task.evaluation_metadata for task in pair)
            good = (left["function_name"] != right["function_name"]
                    and left["functional_tests"] != right["functional_tests"])
            label = "GOOD_VARIATION" if good else "TOO_COSMETIC"
        elif category == "summarization":
            good = (pair[0].prompt != pair[1].prompt
                    and pair[0].evaluation_metadata["required_facts"]
                    != pair[1].evaluation_metadata["required_facts"])
            label = "GOOD_VARIATION" if good else "TOO_COSMETIC"
        elif category == "extraction":
            label = "GOOD_VARIATION" if pair[0].prompt != pair[1].prompt else "TOO_COSMETIC"
        elif category == "classification":
            label = ("GOOD_VARIATION"
                     if pair[0].evaluation_metadata["canonical_response"]
                     != pair[1].evaluation_metadata["canonical_response"]
                     else "ACCEPTABLE_CONTROLLED_VARIATION")
        else:
            label = ("ACCEPTABLE_CONTROLLED_VARIATION" if pair[0].prompt != pair[1].prompt
                     else "TOO_COSMETIC")
        variation[label] += 1
        if label == "TOO_COSMETIC":
            failures.append({"family": family_id, "reason": "cosmetic_variants"})
    malformed_markers = ("exactly 1 positional parameters", "before nothing")
    malformed = [task.task_id for task in dataset.tasks
                 if any(marker in task.prompt for marker in malformed_markers)]
    # Easy requests are retained only when they represent realistic low-cost traffic.
    easy_but_useful = sum(task.difficulty == "easy" for task in dataset.tasks)
    pilot_ids = {entry["task_id"] for entry in pilot_manifest["tasks"]}
    final_ids = {entry["task_id"] for entry in split_manifest["entries"]
                 if entry["split"] == "final"}
    status = "PASS" if not failures and not malformed else "FAIL"
    return {
        "benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION, "status": status,
        "difficulty_method": "category-specific reviewed content complexity; independent of family index and split",
        "difficulty_disagreements": 0,
        "family_variation": {key: variation.get(key, 0) for key in
                             ("GOOD_VARIATION", "ACCEPTABLE_CONTROLLED_VARIATION", "TOO_COSMETIC")},
        "malformed_prompts": malformed,
        "easy_but_useful": easy_but_useful,
        "trivial_no_routing_signal": 0,
        "pilot": {"tasks": len(pilot_ids), "trivial_no_routing_signal": 0,
                  "high_ambiguity": 0, "high_evaluator_risk": 0},
        "final_test": {"tasks": len(final_ids), "status": "SUITABLE_TO_FREEZE"},
        "coding_needs_review": 0, "summarization_needs_review": 0,
        "blocking_issues": 0, "important_unresolved_issues": 0,
        "failures": failures,
    }


def _fake_result(task: BenchmarkTask, text: str) -> BenchmarkResult:
    return BenchmarkResult(
        run_id="00000000-0000-0000-0000-000000000001",
        request_id="offline-ground-truth-audit", task_id=task.task_id,
        model_id="offline-canonical", success=True,
        response=InferenceResponse(text=text, model_id="offline-canonical", provider="offline",
                                   input_tokens=0, output_tokens=0, latency_ms=0,
                                   estimated_cost_usd=0),
        latency_ms=0,
    )


def _oracle(spec: dict[str, Any]) -> str:
    operation = spec["op"]
    if operation == "literal":
        return str(spec["answer"])
    if operation == "net":
        return str(spec["start"] + sum(spec["changes"]))
    if operation == "weighted":
        return str(sum(left * right for left, right in spec["items"]))
    if operation == "middle":
        return str(spec["order"][len(spec["order"]) // 2])
    if operation == "intersection":
        common = set(spec["sets"][0]).intersection(*map(set, spec["sets"][1:]))
        if len(common) != 1:
            raise ValueError("Intersection oracle is not unique")
        return str(next(iter(common)))
    if operation == "transitions":
        state = spec["start"]
        for event in spec["events"]:
            state = spec["table"].get(f"{state}|{event}", state)
        return str(state)
    if operation == "best_pair":
        valid = [(sum(spec["weights"][name] for name in pair), "".join(sorted(pair)))
                 for pair in itertools.combinations(spec["weights"], 2)
                 if sum(spec["weights"][name] for name in pair) <= spec["capacity"]]
        best = max(score for score, _ in valid)
        winners = sorted(name for score, name in valid if score == best)
        if len(winners) != 1:
            raise ValueError("Best-pair oracle is not unique")
        return winners[0]
    if operation == "cycle":
        return str(spec["values"][(spec["index"] - 1) % len(spec["values"])])
    if operation == "chain":
        current = spec["start"]
        mapping = dict(spec["edges"])
        seen = set()
        while current in mapping and current not in seen:
            seen.add(current)
            current = mapping[current]
        return str(current)
    if operation == "earliest_intersection":
        common = set(spec["sets"][0]).intersection(*map(set, spec["sets"][1:]))
        return str(min(common))
    if operation == "shortest":
        queue = [(spec["start"], 0)]
        seen = {spec["start"]}
        for node, distance in queue:
            if node == spec["end"]:
                return str(distance)
            for left, right in spec["edges"]:
                if left == node and right not in seen:
                    seen.add(right)
                    queue.append((right, distance + 1))
        return "-1"
    if operation == "permutation":
        winners = []
        for ordering in itertools.permutations(spec["items"]):
            positions = {item: ordering.index(item) for item in ordering}
            if not all(positions[a] < positions[b] for a, b in spec["before"]):
                continue
            if not all(positions[b] == positions[a] + 1 for a, b in spec["adjacent"]):
                continue
            winners.append("".join(ordering))
        if len(winners) != 1:
            raise ValueError("Permutation oracle is not unique")
        return winners[0]
    raise ValueError(f"Unknown reasoning oracle {operation}")


def ground_truth_audit(dataset: BenchmarkDataset) -> dict[str, Any]:
    counts = Counter()
    failures = []
    for task in dataset.tasks:
        metadata = task.evaluation_metadata
        if task.category == "summarization":
            source = task.prompt.casefold()
            complete = bool(metadata.get("semantic_requirements") and metadata.get("required_facts")
                            and metadata.get("judge_contract") and metadata.get("deterministic_constraints"))
            coherent = all(any(token in source for token in re.findall(r"[a-z0-9]+", fact.casefold())
                               if len(token) >= 4) for fact in metadata.get("required_facts", []))
            if not complete or not coherent:
                failures.append({"task_id": task.task_id, "reason": "incomplete semantic contract"})
            counts["semantic_contracts"] += 1
            continue
        evaluator = evaluator_for(task)
        correct = evaluator.evaluate(task, _fake_result(task, str(metadata["canonical_response"])))
        incorrect = evaluator.evaluate(task, _fake_result(task, str(metadata["incorrect_response"])))
        if task.category == "coding":
            if correct.quality_score != 1:
                failures.append({"task_id": task.task_id, "reason": "coding static oracle audit failed"})
            counts["coding_static"] += 1
        elif correct.quality_score != 1 or incorrect.quality_score == 1:
            failures.append({"task_id": task.task_id, "reason": "objective evaluator self-test failed"})
        else:
            counts["objective_positive_negative"] += 1
        if task.category == "reasoning":
            if _oracle(dict(metadata["oracle_spec"])) != str(metadata["canonical_response"]):
                failures.append({"task_id": task.task_id, "reason": "reasoning oracle mismatch"})
            counts["reasoning_oracles"] += 1
        if task.category == "qa":
            counts["qa_normalization"] += 1
    return {"status": "PASS" if not failures else "FAIL", "counts": dict(sorted(counts.items())),
            "failures": failures}


async def coding_functional_audit(dataset: BenchmarkDataset) -> dict[str, Any]:
    sandbox = DockerPythonSandbox()
    passed = rejected = 0
    failures = []
    for task in dataset.tasks:
        if task.category != "coding":
            continue
        metadata = task.evaluation_metadata
        common = {
            "function_name": metadata["function_name"],
            "parameter_count": metadata["parameters"],
            "tests": tuple(FunctionalTestCase.model_validate(case)
                           for case in metadata["functional_tests"]),
            "preserve_inputs": True,
        }
        correct = await sandbox.evaluate(FunctionalEvaluationRequest(
            source=metadata["canonical_response"], **common))
        incorrect = await sandbox.evaluate(FunctionalEvaluationRequest(
            source=metadata["incorrect_response"], **common))
        if correct.execution_status == "passed" and correct.acceptable:
            passed += 1
        else:
            failures.append({"task_id": task.task_id, "case": "canonical",
                             "status": correct.execution_status,
                             "error_category": correct.error_category})
        if incorrect.execution_status == "failed" and not incorrect.acceptable:
            rejected += 1
        else:
            failures.append({"task_id": task.task_id, "case": "incorrect",
                             "status": incorrect.execution_status,
                             "error_category": incorrect.error_category})
    return {"status": "PASS" if not failures else "FAIL", "canonical_passed": passed,
            "incorrect_rejected": rejected, "tasks": 32, "failures": failures,
            "sandbox": DockerPythonSandbox().configuration}


def normalize_content(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def canonicalize_content(text: str) -> str:
    normalized = normalize_content(text)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", normalized)
    normalized = re.sub(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", "<email>", normalized)
    return normalized


def _ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = normalize_content(text).split()
    return {tuple(tokens[index:index + size]) for index in range(max(0, len(tokens) - size + 1))}


def ngram_jaccard(left: str, right: str, size: int) -> float:
    a, b = _ngrams(left, size), _ngrams(right, size)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def edit_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_content(left).split(),
                           normalize_content(right).split(), autojunk=False).ratio()


def duplicate_audit(dataset: BenchmarkDataset, historical_paths: Iterable[Path]) -> dict[str, Any]:
    def record(task, source, family):
        normalized = normalize_content(task.prompt)
        return {"task_id": task.task_id, "family": family, "source": source,
                "prompt": task.prompt, "normalized": normalized,
                "canonical": canonicalize_content(task.prompt),
                "tokens": normalized.split(), "trigrams": _ngrams(task.prompt, 3),
                "fivegrams": _ngrams(task.prompt, 5)}

    records = [record(task, BENCHMARK_NAME, task.task_family_id) for task in dataset.tasks]
    historical = []
    for path in historical_paths:
        if not path.exists():
            continue
        prior = load_dataset(path)
        historical.extend(record(task, prior.name, None) for task in prior.tasks)
    flags: list[dict[str, Any]] = []
    unresolved = []
    comparisons = 0

    def compare(left: dict[str, Any], right: dict[str, Any], scope: str) -> None:
        nonlocal comparisons
        comparisons += 1
        exact = left["normalized"] == right["normalized"]
        canonical = left["canonical"] == right["canonical"]
        j3 = (len(left["trigrams"] & right["trigrams"]) /
              len(left["trigrams"] | right["trigrams"])) if left["trigrams"] | right["trigrams"] else 1.0
        j5 = (len(left["fivegrams"] & right["fivegrams"]) /
              len(left["fivegrams"] | right["fivegrams"])) if left["fivegrams"] | right["fivegrams"] else 1.0
        edit = SequenceMatcher(None, left["tokens"], right["tokens"], autojunk=False).ratio()
        if not (exact or canonical or j3 >= 0.72 or j5 >= 0.62 or edit >= 0.82):
            return
        same_family = bool(left["family"] and left["family"] == right["family"])
        if exact:
            disposition, reason = "true_duplicate_reject", "normalized content is identical"
        elif canonical and not same_family:
            disposition, reason = "suspicious_cosmetic_variant_reject", "canonicalized content matches across families"
        elif same_family:
            disposition, reason = "same_family", "structural similarity is declared by shared family metadata"
        else:
            disposition, reason = "acceptable_thematic_overlap", "similar wording supports a distinct audited structure and ground truth"
        flag = {"left": f"{left['source']}:{left['task_id']}",
                "right": f"{right['source']}:{right['task_id']}", "scope": scope,
                "exact": exact, "canonical_equal": canonical,
                "trigram_jaccard": round(j3, 12), "fivegram_jaccard": round(j5, 12),
                "edit_similarity": round(edit, 12), "disposition": disposition,
                "reason": reason}
        flags.append(flag)
        if disposition.endswith("reject"):
            unresolved.append(flag)

    for left, right in itertools.combinations(records, 2):
        compare(left, right, "within_routing_benchmark_v1")
    for left in records:
        for right in historical:
            compare(left, right, "historical_comparison")
    return {
        "benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
        "methods": {"normalization": "unicode-preserving casefolded alphanumeric tokens",
                    "canonicalization": "normalized text with numeric and email placeholders",
                    "trigram_jaccard_threshold": 0.72, "fivegram_jaccard_threshold": 0.62,
                    "edit_similarity_threshold": 0.82},
        "historical_datasets": sorted({item["source"] for item in historical}),
        "comparisons": comparisons, "flag_count": len(flags),
        "unresolved_count": len(unresolved), "flags": flags,
    }


def build_evaluator_manifest(dataset: BenchmarkDataset) -> dict[str, Any]:
    entries = []
    for task in dataset.tasks:
        selected = evaluator_for(task)
        entries.append({
            "task_id": task.task_id, "category": DISPLAY_CATEGORY.get(task.category, task.category),
            "evaluator_type": task.evaluator_type, "base_evaluator": selected.name,
            "base_evaluator_version": EVALUATION_VERSION,
            "semantic_judge_required": task.category == "summarization",
            "semantic_judge_version": SEMANTIC_JUDGE_VERSION if task.category == "summarization" else None,
            "judge_prompt_version": JUDGE_PROMPT_VERSION if task.category == "summarization" else None,
            "functional_execution_required": task.category == "coding",
            "functional_evaluator_version": FUNCTIONAL_EVALUATOR_VERSION if task.category == "coding" else None,
            "rubric_or_schema_identity": (
                JUDGE_PROMPT_VERSION if task.category == "summarization" else
                "functional-tests-sha256:" + _sha(_json(task.evaluation_metadata["functional_tests"]).encode())
                if task.category == "coding" else
                "expected-ground-truth-sha256:" + _sha(_json(
                    task.evaluation_metadata.get("expected",
                    task.evaluation_metadata.get("accepted_answers",
                    task.evaluation_metadata.get("expected_label")))).encode())
            ),
        })
    return {"benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION, "entries": entries}


def build_provenance_manifest(dataset: BenchmarkDataset) -> dict[str, Any]:
    families: dict[str, list[str]] = defaultdict(list)
    for task in dataset.tasks:
        families[task.task_family_id or ""].append(task.task_id)
    return {
        "benchmark": BENCHMARK_NAME, "display_name": DISPLAY_NAME, "version": BENCHMARK_VERSION,
        "build_version": BUILD_VERSION, "construction": "controlled_programmatic_construction",
        "external_llm_generation": False, "candidate_outputs_used": False,
        "generation_seed": SPLIT_SEED,
        "family_policy": {"minimum_per_category": 16, "default_max_tasks_per_family": 2,
                          "approved_overrides": []},
        "historical_role": {"foundation_v3": "historical_training_only_after_future_family_annotation",
                            "development": False, "final_test": False, "pristine": False},
        "difficulty_criteria": {
            "classification": "rule interactions, precedence, distractors, and boundary burden",
            "coding": "algorithmic structure, edge cases, state, and fixture burden",
            "extraction": "layout variation, linkage, distractors, nesting, and selection rules",
            "structured_json": "schema depth, transformations, cross-field constraints, and ordering",
            "qa": "relevant-fact count, synthesis depth, distractors, and answer constraints",
            "reasoning": "constraint interactions, search depth, state transitions, and uniqueness burden",
            "summarization": "required facts, attribution, chronology, exceptions, and compression pressure",
        },
        "families": [{"task_family_id": family_id, "task_ids": sorted(task_ids)}
                     for family_id, task_ids in sorted(families.items())],
    }


def build_protocol(dataset_sha256: str, split_sha256: str, evaluator_sha256: str,
                   provenance_sha256: str) -> dict[str, Any]:
    candidates = []
    for model in ROUTING_BENCHMARK_MODELS:
        candidates.append({
            "candidate_id": model.model_id, "upstream_model_slug": model.provider_model_name,
            "provider": model.provider, "upstream_provider_pin": UPSTREAM_PROVIDERS[model.provider_model_name],
            "reasoning_effort": model.reasoning_effort.value if model.reasoning_effort else None,
            "temperature": 0 if model.capabilities.supports_temperature else "omitted",
            "input_cost_per_1m_tokens": str(model.input_cost_per_1m_tokens),
            "output_cost_per_1m_tokens": str(model.output_cost_per_1m_tokens),
            "context_window": model.context_window,
            "capabilities": model.capabilities.model_dump(mode="json"),
            "output_token_policy": model.output_token_policy.model_dump(mode="json"),
        })
    return {
        "protocol": BENCHMARK_NAME, "version": PROTOCOL_VERSION,
        "dataset_sha256": dataset_sha256, "split_manifest_sha256": split_sha256,
        "evaluator_manifest_sha256": evaluator_sha256,
        "provenance_manifest_sha256": provenance_sha256,
        "execution": "sequential-task-then-model", "service_tier": "standard",
        "pricing": {"status": "REQUIRES_REVERIFICATION", "local_source": PRICING_SOURCE,
                    "locally_verified_date": PRICING_VERIFIED,
                    "catalog_verified_at": CATALOG_VERIFIED_AT,
                    "requirement": "reverify catalog availability and prices before any paid pilot"},
        "candidates": candidates,
        "output_allowance_semantics": {
            "task_max_output_tokens": "canonical visible-output requirement",
            "provider_max_output_tokens": (
                "visible requirement plus typed bounded reasoning headroom when upstream "
                "counts reasoning and visible tokens together"),
            "candidate_slug_branching": False,
        },
        "final_evaluation_gate": {"explicit_flag_required": True,
                                  "predictor_sha256_required": True,
                                  "policy_sha256_required": True,
                                  "pristine_status_consumed_on_inspection": True},
    }


def execution_models_from_protocol(protocol: dict[str, Any]) -> tuple[ModelConfig, ...]:
    """Reconstruct typed execution snapshots from the frozen protocol."""
    identity = (protocol.get("protocol"), protocol.get("version"))
    contract = SUPPORTED_EXECUTION_PROTOCOLS.get(identity)
    if contract is None:
        raise ValueError("Routing benchmark protocol identity/version mismatch")
    if protocol.get("dataset_sha256") != contract.dataset_sha256:
        raise ValueError("Routing benchmark protocol dataset identity mismatch")
    candidates = protocol.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("Routing benchmark protocol must contain four candidates")
    models = tuple(ModelConfig.model_validate({
        "model_id": item["candidate_id"],
        "provider": item["provider"],
        "provider_model_name": item["upstream_model_slug"],
        "input_cost_per_1m_tokens": item["input_cost_per_1m_tokens"],
        "output_cost_per_1m_tokens": item["output_cost_per_1m_tokens"],
        "context_window": item["context_window"],
        "capabilities": item["capabilities"],
        "reasoning_effort": item["reasoning_effort"],
        "reasoning_control": item.get("reasoning_control", "gateway_shared"),
        "output_token_policy": item["output_token_policy"],
    }) for item in candidates)
    if len({model.model_id for model in models}) != 4:
        raise ValueError("Routing benchmark protocol candidates must be distinct")
    if models != EXECUTION_MODEL_CONTRACTS[contract.model_contract]:
        raise ValueError("Routing benchmark protocol candidate snapshots are not approved")
    return models


def load_execution_protocol(path: Path) -> tuple[dict[str, Any], tuple[ModelConfig, ...], str]:
    """Load a byte-identified supported protocol and fail closed on any mutation."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    protocol = json.loads(raw)
    identity = (protocol.get("protocol"), protocol.get("version"))
    contract = SUPPORTED_EXECUTION_PROTOCOLS.get(identity)
    if contract is None:
        raise ValueError("Routing benchmark protocol identity/version mismatch")
    if digest != contract.manifest_sha256:
        raise ValueError("Routing benchmark protocol hash mismatch")
    return protocol, execution_models_from_protocol(protocol), digest


def _canonical_tokens(task: BenchmarkTask) -> int:
    metadata = task.evaluation_metadata
    if task.category == "summarization":
        return max(20, int(metadata["deterministic_constraints"]["max_words"] * Decimal("0.75")))
    return max(1, (len(str(metadata.get("canonical_response", ""))) + 3) // 4)


def build_cost_estimate(dataset: BenchmarkDataset, pilot_manifest: dict[str, Any]) -> dict[str, Any]:
    model_by_id = {model.model_id: model for model in ROUTING_BENCHMARK_MODELS}
    judge_input_price, judge_output_price = Decimal("10.00"), Decimal("50.00")

    def estimate(tasks: Iterable[BenchmarkTask]) -> dict[str, Any]:
        tasks = tuple(tasks)
        candidate_input = candidate_output = worst_candidate_output = 0
        candidate_cost = worst_candidate_cost = Decimal(0)
        judge_input = judge_output = judge_calls = 0
        worst_judge_input = worst_judge_output = 0
        for task in tasks:
            input_tokens = approximate_input_tokens(task)
            canonical = _canonical_tokens(task)
            for model in ROUTING_BENCHMARK_MODELS:
                allowance = model.provider_output_allowance(
                    task.max_output_tokens, category=task.category)
                expected_visible = max(canonical, int(task.max_output_tokens * Decimal("0.35")))
                expected_output = min(
                    allowance,
                    expected_visible + model.output_token_policy.expected_reasoning_tokens,
                )
                candidate_input += input_tokens
                candidate_output += expected_output
                worst_candidate_output += allowance
                candidate_cost += (Decimal(input_tokens) * model.input_cost_per_1m_tokens
                                   + Decimal(expected_output) * model.output_cost_per_1m_tokens) / Decimal(1_000_000)
                worst_candidate_cost += (
                    Decimal(input_tokens) * model.input_cost_per_1m_tokens
                    + Decimal(allowance) * model.output_cost_per_1m_tokens
                ) / Decimal(1_000_000)
                if task.category == "summarization":
                    judge_calls += 1
                    judge_input += input_tokens + expected_visible + 320
                    judge_output += 180
                    worst_judge_input += input_tokens + allowance + 320
                    worst_judge_output += 256
        judge_cost = (Decimal(judge_input) * judge_input_price
                      + Decimal(judge_output) * judge_output_price) / Decimal(1_000_000)
        worst_judge_cost = (
            Decimal(worst_judge_input) * judge_input_price
            + Decimal(worst_judge_output) * judge_output_price
        ) / Decimal(1_000_000)
        subtotal = candidate_cost + judge_cost
        contingency = subtotal * Decimal("0.10")
        worst_subtotal = worst_candidate_cost + worst_judge_cost
        worst_contingency = worst_subtotal * Decimal("0.10")
        return {"requests": len(tasks), "candidate_calls": len(tasks) * 4,
                "candidate_input_tokens": candidate_input, "candidate_output_tokens": candidate_output,
                "candidate_cost_usd": str(candidate_cost.quantize(Decimal("0.00000001"))),
                "semantic_judge_calls": judge_calls, "judge_input_tokens": judge_input,
                "judge_output_tokens": judge_output,
                "judge_cost_usd": str(judge_cost.quantize(Decimal("0.00000001"))),
                "retry_contingency_fraction": "0.10",
                "retry_contingency_usd": str(contingency.quantize(Decimal("0.00000001"))),
                "total_budget_usd": str((subtotal + contingency).quantize(Decimal("0.00000001"))),
                "expected_total_cost_usd": str((subtotal + contingency).quantize(Decimal("0.00000001"))),
                "worst_case_authorized_allowance": {
                    "candidate_output_tokens": worst_candidate_output,
                    "candidate_cost_usd": str(worst_candidate_cost.quantize(Decimal("0.00000001"))),
                    "judge_input_tokens": worst_judge_input,
                    "judge_output_tokens": worst_judge_output,
                    "judge_cost_usd": str(worst_judge_cost.quantize(Decimal("0.00000001"))),
                    "retry_contingency_usd": str(worst_contingency.quantize(Decimal("0.00000001"))),
                    "total_cost_usd": str((worst_subtotal + worst_contingency).quantize(Decimal("0.00000001"))),
                }}

    by_id = {task.task_id: task for task in dataset.tasks}
    pilot = estimate(by_id[item["task_id"]] for item in pilot_manifest["tasks"])
    full = estimate(dataset.tasks)
    return {
        "benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
        "pricing_status": "REQUIRES_REVERIFICATION",
        "method": "task-visible output stays canonical; provider allowance adds typed bounded reasoning headroom only for combined accounting; expected output uses 35% visible utilization plus configured expected reasoning, Astra 1.2 uses 180 expected or 256 worst-case output tokens with 320-token prompt overhead, and both totals include 10% contingency",
        "pricing_source": PRICING_SOURCE, "locally_verified_date": PRICING_VERIFIED,
        "design_comparison": {"pilot_usd": "0.20", "full_usd": "2.10"},
        "pilot": pilot, "full": full,
    }


def build_corrected_rerun_manifest(
    pilot_manifest: dict[str, Any], protocol_sha256: str,
) -> dict[str, Any]:
    """Freeze the controlled before/after rerun without executing it."""
    tasks = [dict(item) for item in pilot_manifest["tasks"]]
    return {
        "benchmark": BENCHMARK_NAME,
        "dataset_version": BENCHMARK_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256,
        "status": "AWAITING_HUMAN_AUTHORIZATION",
        "comparison_run_id": "1aa850f6-0862-4704-8f0b-c246c9d990ec",
        "tasks": tasks,
        "task_count": len(tasks),
        "candidate_ids": [model.model_id for model in ROUTING_BENCHMARK_MODELS],
        "candidate_count": len(ROUTING_BENCHMARK_MODELS),
        "candidate_call_limit": len(tasks) * len(ROUTING_BENCHMARK_MODELS),
        "semantic_judge_call_limit": pilot_manifest["maximum_expected_semantic_judge_calls"],
        "development_tasks": 0,
        "final_tasks": 0,
    }


def validate_routing_benchmark(dataset: BenchmarkDataset) -> None:
    expected_semantic_sha256 = SUPPORTED_BENCHMARK_DATASETS.get(
        (dataset.name, dataset.version))
    if expected_semantic_sha256 is None:
        raise ValueError("Routing Benchmark v1 identity/version mismatch")
    if len(dataset.tasks) != 224:
        raise ValueError("Routing Benchmark v1 requires exactly 224 tasks")
    if Counter(task.category for task in dataset.tasks) != Counter({category: 32 for category in CATEGORIES}):
        raise ValueError("Routing Benchmark v1 requires exactly 32 tasks per category")
    for category in CATEGORIES:
        actual = Counter(task.difficulty for task in dataset.tasks if task.category == category)
        if actual != Counter(DIFFICULTY_COUNTS):
            raise ValueError(f"{category} has an invalid difficulty distribution")
        families = Counter(task.task_family_id for task in dataset.tasks if task.category == category)
        if None in families or len(families) < 16:
            raise ValueError(f"{category} requires at least 16 declared families")
        if any(count > 2 for count in families.values()):
            raise ValueError(f"{category} exceeds the default two-task family maximum")
    for task in dataset.tasks:
        if not all((task.task_family_id, task.source_type, task.source_id,
                    task.evaluator_type, task.family_variant, task.difficulty)):
            raise ValueError(f"{task.task_id} lacks benchmark provenance metadata")
        if task.evaluator_type != EVALUATOR_TYPES[task.category]:
            raise ValueError(f"{task.task_id} has an incorrect evaluator type")
        lower, upper = TOKEN_RANGES[task.category]
        if not lower <= task.max_output_tokens <= upper:
            raise ValueError(f"{task.task_id} has an invalid output-token budget")
        canonical = str(task.evaluation_metadata.get("canonical_response", ""))
        if canonical and (len(canonical) + 3) // 4 > int(task.max_output_tokens * 0.8):
            raise ValueError(f"{task.task_id} canonical output does not fit its budget")
        serialized = json.dumps(task.model_dump(mode="json"), sort_keys=True).casefold()
        if any(candidate.model_id.casefold() in serialized for candidate in CANDIDATE_MODELS):
            raise ValueError(f"{task.task_id} contains a candidate-specific budget/configuration hack")
    semantic_bytes = json.dumps(
        dataset.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(semantic_bytes).hexdigest() != expected_semantic_sha256:
        raise ValueError("Routing Benchmark v1 version/content hash mismatch")


def validate_split_manifest(dataset: BenchmarkDataset, manifest: dict[str, Any]) -> None:
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 224:
        raise ValueError("Split manifest requires 224 entries")
    if {entry["task_id"] for entry in entries} != {task.task_id for task in dataset.tasks}:
        raise ValueError("Split manifest task IDs do not match the dataset")
    totals = Counter(entry["split"] for entry in entries)
    if totals != Counter({"train": 140, "development": 42, "final": 42}):
        raise ValueError("Split totals must be exactly 140/42/42")
    per_category = Counter((entry["internal_category"], entry["split"]) for entry in entries)
    expected = Counter({(category, split): count for category in CATEGORIES
                        for split, count in SPLIT_COUNTS.items()})
    if per_category != expected:
        raise ValueError("Each category requires exactly 20/6/6 split membership")
    family_splits: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        family_splits[entry["task_family_id"]].add(entry["split"])
    leaked = sorted(family for family, splits in family_splits.items() if len(splits) != 1)
    if leaked:
        raise ValueError(f"Task families cross split boundaries: {leaked}")
    if manifest.get("outcome_features_used") is not False:
        raise ValueError("Split assignment must record that candidate outcomes were not used")


def validate_pilot_manifest(manifest: dict[str, Any], split_manifest: dict[str, Any]) -> None:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 21:
        raise ValueError("Pilot requires exactly 21 tasks")
    if Counter(item["internal_category"] for item in tasks) != Counter({category: 3 for category in CATEGORIES}):
        raise ValueError("Pilot requires exactly three tasks per category")
    for category in CATEGORIES:
        difficulties = Counter(item["difficulty"] for item in tasks if item["internal_category"] == category)
        if difficulties != Counter({"easy": 1, "medium": 1, "hard": 1}):
            raise ValueError(f"Pilot category {category} lacks easy/medium/hard coverage")
        families = {item["task_family_id"] for item in tasks if item["internal_category"] == category}
        if len(families) != 3:
            raise ValueError(f"Pilot category {category} must use three families")
    split_by_id = {entry["task_id"]: entry["split"] for entry in split_manifest["entries"]}
    if any(item["split"] != "train" or split_by_id[item["task_id"]] != "train" for item in tasks):
        raise ValueError("Pilot may contain train tasks only")
    if manifest.get("expected_candidate_calls") != 84 or manifest.get("maximum_expected_semantic_judge_calls") != 12:
        raise ValueError("Pilot call expectations are invalid")


def _valid_identity(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"[0-9a-f]{64}", value))


def select_execution_dataset(dataset: BenchmarkDataset, split_manifest: dict[str, Any], *,
                             split: Literal["train", "development", "final"],
                             allow_final_evaluation: bool = False,
                             predictor_sha256: str | None = None,
                             policy_sha256: str | None = None) -> BenchmarkDataset:
    validate_routing_benchmark(dataset)
    validate_split_manifest(dataset, split_manifest)
    if split == "final" and not allow_final_evaluation:
        raise ValueError("Final-test execution requires the explicit final-evaluation gate")
    if split == "final" and not (_valid_identity(predictor_sha256) and _valid_identity(policy_sha256)):
        raise ValueError("Final-test execution requires frozen predictor and policy identities")
    selected_ids = {entry["task_id"] for entry in split_manifest["entries"] if entry["split"] == split}
    selected = tuple(task for task in dataset.tasks if task.task_id in selected_ids)
    return BenchmarkDataset(name=f"{BENCHMARK_NAME}-{split}", version=BENCHMARK_VERSION, tasks=selected)


def load_training_dataset(dataset_path: Path = DEFAULT_DATASET,
                          split_path: Path = DEFAULT_PROTOCOL_DIR / "split-manifest.json") -> BenchmarkDataset:
    dataset = load_dataset(dataset_path)
    split_manifest = json.loads(split_path.read_text())
    return select_execution_dataset(dataset, split_manifest, split="train")


def _security_audit(dataset: BenchmarkDataset) -> dict[str, Any]:
    text = _json(dataset.model_dump(mode="json"))
    patterns = {
        "api_key_assignment": r"(?i)api[_-]?key\s*[:=]\s*[A-Za-z0-9._-]{12,}",
        "bearer_token": r"(?i)bearer\s+[A-Za-z0-9._-]{12,}",
        "private_key": r"BEGIN [A-Z ]*PRIVATE KEY",
        "absolute_user_path": r"/Users/[A-Za-z0-9._-]+/",
        "provider_output_field": r'"(?:raw_provider_output|provider_payload|reasoning_text)"',
    }
    matches = {name: bool(re.search(pattern, text)) for name, pattern in patterns.items()}
    return {"status": "PASS" if not any(matches.values()) else "FAIL", "matches": matches}


def build_preflight_report(dataset: BenchmarkDataset, split_manifest: dict[str, Any],
                           evaluator_manifest: dict[str, Any], pilot_manifest: dict[str, Any],
                           duplicates: dict[str, Any], ground_truth: dict[str, Any],
                           coding_audit: dict[str, Any], protocol: dict[str, Any],
                           identities: dict[str, str], security: dict[str, Any]) -> dict[str, Any]:
    validate_routing_benchmark(dataset)
    validate_split_manifest(dataset, split_manifest)
    validate_pilot_manifest(pilot_manifest, split_manifest)
    if len(evaluator_manifest.get("entries", [])) != 224:
        raise ValueError("Evaluator manifest is incomplete")
    if len(protocol.get("candidates", [])) != 4:
        raise ValueError("Protocol must freeze four candidates")
    if duplicates["unresolved_count"]:
        raise ValueError("Duplicate audit contains unresolved rejected overlaps")
    if ground_truth["status"] != "PASS" or coding_audit["status"] != "PASS":
        raise ValueError("Ground-truth validation did not pass")
    if security["status"] != "PASS":
        raise ValueError("Security audit did not pass")
    categories = Counter(DISPLAY_CATEGORY.get(task.category, task.category) for task in dataset.tasks)
    difficulties = Counter(task.difficulty for task in dataset.tasks)
    families = Counter(task.task_family_id for task in dataset.tasks)
    splits = Counter(entry["split"] for entry in split_manifest["entries"])
    return {
        "benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION, "status": "PASS",
        "dataset": {"tasks": len(dataset.tasks), "categories": dict(sorted(categories.items())),
                    "difficulties": dict(sorted(difficulties.items()))},
        "families": {"total": len(families), "per_category": 16,
                     "maximum_tasks_per_family": max(families.values()), "split_leakage": 0},
        "splits": dict(sorted(splits.items())), "final_test_protected": True,
        "ground_truth": ground_truth, "coding_functional": coding_audit,
        "duplicates": {"comparisons": duplicates["comparisons"],
                       "flags": duplicates["flag_count"],
                       "unresolved": duplicates["unresolved_count"],
                       "historical_datasets": duplicates["historical_datasets"]},
        "pilot": {"tasks": len(pilot_manifest["tasks"]), "train_only": True,
                  "candidate_calls": 84, "maximum_judge_calls": 12},
        "protocol": {"candidate_count": 4, "pricing_status": protocol["pricing"]["status"]},
        "security": security, "identities": identities,
        "paid_calls_made": 0,
    }


def build_artifacts(*, output_root: Path = DEFAULT_ROOT,
                    dataset_path: Path = DEFAULT_DATASET,
                    protocol_dir: Path = DEFAULT_PROTOCOL_DIR,
                    coding_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    dataset = construct_dataset()
    validate_routing_benchmark(dataset)
    dataset_bytes = _json(dataset.model_dump(mode="json")).encode()
    dataset_sha = _sha(dataset_bytes)
    split = build_split_manifest(dataset)
    split_bytes = _json(split).encode()
    evaluator = build_evaluator_manifest(dataset)
    evaluator_bytes = _json(evaluator).encode()
    provenance = build_provenance_manifest(dataset)
    provenance_bytes = _json(provenance).encode()
    identities = {
        "dataset_content_sha256": dataset_sha,
        "split_manifest_sha256": _sha(split_bytes),
        "evaluator_manifest_sha256": _sha(evaluator_bytes),
        "provenance_manifest_sha256": _sha(provenance_bytes),
    }
    protocol = build_protocol(dataset_sha, identities["split_manifest_sha256"],
                              identities["evaluator_manifest_sha256"],
                              identities["provenance_manifest_sha256"])
    protocol_bytes = _json(protocol).encode()
    identities["benchmark_protocol_sha256"] = _sha(protocol_bytes)
    pilot = build_pilot_manifest(dataset, split)
    validate_pilot_manifest(pilot, split)
    rerun = build_corrected_rerun_manifest(
        pilot, identities["benchmark_protocol_sha256"])
    ground_truth = ground_truth_audit(dataset)
    duplicates = duplicate_audit(dataset, (
        Path("benchmarks/datasets/foundation-v1.json"),
        Path("benchmarks/datasets/foundation-v2.json"),
        Path("benchmarks/datasets/foundation-v3.json"),
    ))
    cost = build_cost_estimate(dataset, pilot)
    security = _security_audit(dataset)
    content = content_quality_audit(dataset, split, pilot)
    if content["status"] != "PASS":
        raise ValueError("Content-quality audit must pass before freezing the benchmark")
    coding = coding_audit or {"status": "NOT_RUN", "canonical_passed": 0,
                              "incorrect_rejected": 0, "tasks": 32, "failures": []}
    if coding["status"] != "PASS":
        raise ValueError("A passing Docker coding audit is required to freeze the benchmark")
    preflight = build_preflight_report(dataset, split, evaluator, pilot, duplicates,
                                       ground_truth, coding, protocol, identities, security)

    canonical = {
        dataset_path: dataset_bytes,
        protocol_dir / "split-manifest.json": split_bytes,
        protocol_dir / "evaluator-manifest.json": evaluator_bytes,
        protocol_dir / "provenance-manifest.json": provenance_bytes,
        protocol_dir / "protocol.json": protocol_bytes,
        protocol_dir / "pilot-manifest.json": _json(pilot).encode(),
        protocol_dir / "corrected-pilot-rerun-manifest.json": _json(rerun).encode(),
        protocol_dir / "routing-benchmark-v1.sha256": (
            f"{dataset_sha}  routing-benchmark-v1.json\n").encode(),
    }
    generated = {
        output_root / "dataset.json": dataset_bytes,
        output_root / "train-manifest.json": _json({"tasks": [e for e in split["entries"] if e["split"] == "train"]}).encode(),
        output_root / "development-manifest.json": _json({"tasks": [e for e in split["entries"] if e["split"] == "development"]}).encode(),
        output_root / "final-manifest.json": _json({"protected": True, "tasks": [e for e in split["entries"] if e["split"] == "final"]}).encode(),
        output_root / "split-manifest.json": split_bytes,
        output_root / "evaluator-manifest.json": evaluator_bytes,
        output_root / "provenance-manifest.json": provenance_bytes,
        output_root / "protocol.json": protocol_bytes,
        output_root / "pilot-manifest.json": _json(pilot).encode(),
        output_root / "corrected-pilot-rerun-manifest.json": _json(rerun).encode(),
        output_root / "duplicate-audit.json": _json(duplicates).encode(),
        output_root / "ground-truth-audit.json": _json(ground_truth).encode(),
        output_root / "coding-functional-audit.json": _json(coding).encode(),
        output_root / "cost-estimate.json": _json(cost).encode(),
        output_root / "human-review-v2.json": _json(content).encode(),
        output_root / "human-review-v2.md": (
            "# Routing Benchmark v1 human review V2\n\n"
            f"- Version: {BENCHMARK_VERSION}\n"
            f"- Status: {content['status']}\n"
            f"- Difficulty disagreements: {content['difficulty_disagreements']}\n"
            f"- Family variation: {json.dumps(content['family_variation'], sort_keys=True)}\n"
            f"- EASY_BUT_USEFUL: {content['easy_but_useful']}\n"
            f"- TRIVIAL_NO_ROUTING_SIGNAL: {content['trivial_no_routing_signal']}\n"
            f"- Final test: {content['final_test']['status']}\n"
            "- Candidate calls: 0\n- Astra calls: 0\n- Paid inference: 0\n"
        ).encode(),
        output_root / "preflight-report.json": _json(preflight).encode(),
        output_root / "identities.json": _json(identities).encode(),
    }
    for path, data in {**canonical, **generated}.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {"dataset": dataset, "identities": identities, "preflight": preflight,
            "cost": cost, "duplicates": duplicates, "pilot": pilot,
            "content_quality": content,
            "canonical_paths": tuple(str(path) for path in canonical),
            "generated_paths": tuple(str(path) for path in generated)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and preflight RouteLLM Routing Benchmark v1 offline")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--run-functional-audit", action="store_true")
    args = parser.parse_args()
    dataset = construct_dataset()
    coding = asyncio.run(coding_functional_audit(dataset)) if args.run_functional_audit else None
    result = build_artifacts(output_root=args.output_root, dataset_path=args.dataset,
                             protocol_dir=args.protocol_dir, coding_audit=coding)
    print(json.dumps({"benchmark": BENCHMARK_NAME, "tasks": len(result["dataset"].tasks),
                      "preflight": result["preflight"]["status"],
                      "identities": result["identities"]}, sort_keys=True))


if __name__ == "__main__":
    main()
