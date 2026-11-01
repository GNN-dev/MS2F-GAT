"""CLI orchestration for the three-dataset, two-representation LLM benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .api import (
    RollingRateLimiter,
    build_request,
    common_max_input_tokens,
    enabled_providers,
    execute_request,
    request_sha256,
    utc_now,
)
from .data import (
    PROJECT_ROOT,
    REPRESENTATIONS,
    SYSTEM_PROMPT,
    canonical_json,
    config_sha256,
    inventory_dataset,
    load_dataset_bundle,
    load_tokenizer,
    project_path,
    read_jsonl,
    sha256_text,
    user_prompt,
    write_json,
    write_jsonl,
)
from .metrics import (
    FIVE_METRICS,
    bootstrap_mean_confidence_intervals,
    multilabel_metrics,
    per_label_metrics,
)




def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict zero-shot LLM benchmark on UML-10, ECORE-10, and V12"
    )
    parser.add_argument("--mode", choices=("inventory", "prepare", "verify", "preflight", "run", "report"), default="inventory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--providers", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--execute-paid-api-calls", action="store_true")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    config = json.loads(resolved.read_text(encoding="utf-8"))
    config["_config_path"] = str(resolved.resolve())
    return config


def output_dir_for(config: dict[str, Any], override: Path | None) -> Path:
    path = override or Path(str(config["output_dir"]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_dataset_names(config: dict[str, Any], requested: Sequence[str] | None) -> list[str]:
    names = list(config["datasets"])
    selected = list(requested) if requested else names
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return selected


def build_inventory(
    config: dict[str, Any], dataset_names: Sequence[str], output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = load_tokenizer(config)
    rows: list[dict[str, Any]] = []
    audits = {}
    for name in dataset_names:
        bundle = load_dataset_bundle(name, config["datasets"][name])
        try:
            dataset_rows, audit = inventory_dataset(bundle, tokenizer)
            rows.extend(dataset_rows)
            audits[name] = audit
        finally:
            bundle.close()
    write_jsonl(output_dir / "candidate_inventory.jsonl", rows)
    write_json(
        output_dir / "inventory_audit.json",
        {
            "status": "PASS",
            "datasets": audits,
            "tokenizer": config["protocol"]["tokenizer"],
            "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
            "config_sha256": config_sha256(config),
            "api_calls_made": 0,
        },
    )
    return rows, audits


def prepare(config: dict[str, Any], dataset_names: Sequence[str], output_dir: Path) -> None:
    providers = enabled_providers(config)
    common_limit, provider_limits = common_max_input_tokens(config)
    inventory, audits = build_inventory(config, dataset_names, output_dir)
    manifest = []
    excluded = []
    for row in inventory:
        maximum = max(int(row[representation]["prompt_tokens"]) for representation in REPRESENTATIONS)
        record = {
            "dataset": row["dataset"],
            "model_id": row["model_id"],
            "gold_label_ids": row["gold_label_ids"],
            "raw_artifact": row["raw_artifact"],
            "full_graph_json": row["full_graph_json"],
        }
        if maximum <= common_limit:
            manifest.append(record)
        else:
            excluded.append(
                {
                    **record,
                    "reason": "paired_complete_input_exceeds_common_provider_budget",
                    "common_max_input_tokens": common_limit,
                }
            )
    for name in dataset_names:
        eligible = [row for row in manifest if row["dataset"] == name]
        if not eligible:
            raise ValueError(f"No paired context-eligible samples for {name}")
        full_labels = {item["id"] for item in audits[name]["label_catalog"]}
        supported = {label for row in eligible for label in row["gold_label_ids"]}
        if supported != full_labels:
            raise ValueError(f"Paired filtering removed all positives for {name}: {sorted(full_labels - supported)}")

    write_jsonl(output_dir / "input_manifest.jsonl", manifest)
    write_json(output_dir / "excluded_context_overflow.json", excluded)
    manifest_hash = sha256_file(output_dir / "input_manifest.jsonl")
    run_manifest = {
        "protocol_name": config["protocol_name"],
        "protocol_version": config["protocol_version"],
        "config_sha256": config_sha256(config),
        "input_manifest_sha256": manifest_hash,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "enabled_providers": {
            name: {
                "api_style": provider["api_style"],
                "base_url": provider["base_url"],
                "model": provider["model"],
                "context_window_tokens": provider["context_window_tokens"],
                "max_output_tokens": provider["max_output_tokens"],
                "max_tokens_parameter": provider.get(
                    "max_tokens_parameter", "max_tokens"
                ),
                "prompt_token_safety_multiplier": float(
                    provider.get("prompt_token_safety_multiplier", 1.0)
                ),
                "json_object_mode": bool(provider.get("json_object_mode", True)),
                "omit_temperature": bool(provider.get("omit_temperature", False)),
                "extra_request": dict(provider.get("extra_request", {})),
            }
            for name, provider in providers.items()
        },
        "provider_max_input_tokens": provider_limits,
        "common_max_input_tokens": common_limit,
        "paired_examples": {
            name: sum(row["dataset"] == name for row in manifest) for name in dataset_names
        },
        "excluded_examples": {
            name: sum(row["dataset"] == name for row in excluded) for name in dataset_names
        },
        "dataset_label_notes": {
            name: config["datasets"][name].get("reference_label_note", "")
            for name in dataset_names
        },
        "repeats": int(config["protocol"]["repeats"]),
        "representations": list(REPRESENTATIONS),
        "metrics": list(FIVE_METRICS),
        "temperature": float(config["protocol"]["temperature"]),
        "token_estimator": config["protocol"]["tokenizer"],
        "oversize_policy": "exclude_from_both_representations_before_api_calls",
        "chunking": "disabled; multi-request chunk aggregation is a different protocol",
        "api_calls_made": 0,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    packages = {}
    for package in ("openai", "tiktoken", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    write_json(
        output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "api_keys_recorded": False,
        },
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))


def verify_prepared(config: dict[str, Any], output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = output_dir / "input_manifest.jsonl"
    run_manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file() or not run_manifest_path.is_file():
        raise FileNotFoundError("Run prepare after provider configuration")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest["config_sha256"] != config_sha256(config):
        raise ValueError("Configuration changed after prepare; rebuild the frozen manifest")
    if run_manifest["input_manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Frozen input manifest hash mismatch")
    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise ValueError("Frozen input manifest is empty")
    return manifest, run_manifest


def response_path(
    output_dir: Path, provider: str, dataset: str, representation: str, repeat: int
) -> Path:
    return output_dir / "responses" / provider / dataset / representation / f"repeat_{repeat}.jsonl"


def latest_response_map(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        if row.get("model_id"):
            result[str(row["model_id"])] = row
    return result


def reconstruct_prompt(
    config: dict[str, Any], dataset: str, model_id: str, representation: str
) -> tuple[str, list[str]]:
    bundle = load_dataset_bundle(dataset, config["datasets"][dataset])
    try:
        payload, _ = bundle.input_payload(model_id, representation)
        return user_prompt(representation, payload, bundle.label_specs), bundle.label_ids
    finally:
        bundle.close()


def preflight(
    config: dict[str, Any],
    provider_names: Sequence[str],
    output_dir: Path,
    execute_paid: bool,
) -> None:
    if not execute_paid:
        raise ValueError("Preflight calls an external paid API; add --execute-paid-api-calls")
    providers = enabled_providers(config)
    results = []
    attempts_path = output_dir / "preflight_attempts.jsonl"
    for name in provider_names:
        provider = providers[name]
        prompt, labels = preflight_prompt()
        request = build_request(provider, prompt, float(config["protocol"]["temperature"]))
        result = execute_request(
            provider_name=name,
            provider=provider,
            request=request,
            request_hash=request_sha256(name, provider, request),
            label_ids=labels,
        )
        record = {"provider": name, **result}
        results.append(record)
        attempts_path.parent.mkdir(parents=True, exist_ok=True)
        with attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
        print(json.dumps(record, ensure_ascii=False))
    attempts = read_jsonl(attempts_path)
    latest = {}
    for attempt in attempts:
        latest[str(attempt["provider"])] = attempt
    write_json(
        output_dir / "preflight.json",
        {
            "latest_results": latest,
            "attempt_count": len(attempts),
            "all_latest_ok": all(
                row.get("status") == "ok" for row in latest.values()
            ),
        },
    )


def preflight_prompt() -> tuple[str, list[str]]:
    return (
        "Connectivity and JSON-format check. Label catalog:\n- L001 = test\n"
        'Return exactly {"decisions":{"L001":0}}.',
        ["L001"],
    )


def verify_preflight_gate(
    config: dict[str, Any],
    output_dir: Path,
    provider_names: Sequence[str],
    providers: dict[str, dict[str, Any]],
) -> None:
    path = output_dir / "preflight.json"
    if not path.is_file():
        raise FileNotFoundError("Run preflight successfully before formal API execution")
    summary = json.loads(path.read_text(encoding="utf-8"))
    latest = dict(summary.get("latest_results", {}))
    prompt, _ = preflight_prompt()
    failures = []
    for name in provider_names:
        provider = providers[name]
        row = latest.get(name, {})
        request = build_request(
            provider, prompt, float(config["protocol"]["temperature"])
        )
        expected_hash = request_sha256(name, provider, request)
        if (
            row.get("status") != "ok"
            or row.get("configured_model") != provider["model"]
            or row.get("actual_model") != provider["model"]
            or row.get("request_sha256") != expected_hash
        ):
            failures.append(name)
    if failures:
        raise ValueError(
            "Missing, failed, stale, or model-mismatched preflight results: "
            + ", ".join(failures)
        )


def circuit_breaker_reason(
    result: dict[str, Any],
    consecutive_api_failures: int,
    protocol: dict[str, Any],
) -> str | None:
    """Return a provider-level halt reason without retrying the failed request."""
    if result.get("status") != "api_failed":
        return None
    category = str(result.get("failure_category", "OTHER_API_FAILURE"))
    terminal = set(
        protocol.get(
            "terminal_failure_categories",
            ("INSUFFICIENT_BALANCE_OR_QUOTA", "AUTHENTICATION", "RATE_LIMIT"),
        )
    )
    if category in terminal:
        return f"terminal_failure:{category}"
    limit = int(protocol.get("consecutive_api_failure_limit", 3))
    if limit > 0 and consecutive_api_failures >= limit:
        return f"consecutive_api_failures:{consecutive_api_failures}"
    return None


def append_run_event(output_dir: Path, event: dict[str, Any]) -> None:
    path = output_dir / "run_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def run_api(
    config: dict[str, Any],
    output_dir: Path,
    requested_providers: Sequence[str] | None,
    dataset_names: Sequence[str],
    workers_override: int | None,
    retry_failures: bool,
    execute_paid: bool,
) -> None:
    if not execute_paid:
        raise ValueError("Run calls paid external APIs; add --execute-paid-api-calls")
    manifest, _ = verify_prepared(config, output_dir)
    providers = enabled_providers(config)
    provider_names = list(requested_providers) if requested_providers else list(providers)
    unknown = sorted(set(provider_names) - set(providers))
    if unknown:
        raise ValueError(f"Requested providers are not enabled: {unknown}")
    verify_preflight_gate(config, output_dir, provider_names, providers)
    repeats = int(config["protocol"]["repeats"])
    temperature = float(config["protocol"]["temperature"])

    for provider_name in provider_names:
        provider_halted = False
        consecutive_api_failures = 0
        provider = providers[provider_name]
        limiter = RollingRateLimiter(
            int(provider["tokens_per_minute"]) if provider.get("tokens_per_minute") else None,
            int(provider["requests_per_minute"]) if provider.get("requests_per_minute") else None,
        )
        workers = workers_override or int(provider.get("workers", 1))
        if workers < 1:
            raise ValueError("workers must be at least 1")
        for dataset in dataset_names:
            if provider_halted:
                break
            dataset_rows = [row for row in manifest if row["dataset"] == dataset]
            bundle = load_dataset_bundle(dataset, config["datasets"][dataset])
            try:
                input_lock = threading.Lock()
                for representation in REPRESENTATIONS:
                    if provider_halted:
                        break
                    for repeat in range(1, repeats + 1):
                        if provider_halted:
                            break
                        path = response_path(output_dir, provider_name, dataset, representation, repeat)
                        cached = latest_response_map(path)
                        tasks = []
                        for row in dataset_rows:
                            existing = cached.get(str(row["model_id"]))
                            if existing and existing.get("status") in {"ok", "invalid_response"}:
                                continue
                            if existing and existing.get("status") == "api_failed" and not retry_failures:
                                continue
                            tasks.append(row)
                        if not tasks:
                            print(f"{provider_name}/{dataset}/{representation}/repeat_{repeat}: nothing to run", flush=True)
                            continue

                        path.parent.mkdir(parents=True, exist_ok=True)
                        write_lock = threading.Lock()

                        def call_one(row: dict[str, Any]) -> dict[str, Any]:
                            model_id = str(row["model_id"])
                            with input_lock:
                                payload, _ = bundle.input_payload(model_id, representation)
                            prompt = user_prompt(representation, payload, bundle.label_specs)
                            expected_prompt_hash = row[representation]["prompt_sha256"]
                            if sha256_text(SYSTEM_PROMPT + "\n" + prompt) != expected_prompt_hash:
                                raise ValueError(f"Prompt reconstruction mismatch for {dataset}/{model_id}/{representation}")
                            request = build_request(provider, prompt, temperature)
                            limiter.acquire(
                                math.ceil(
                                    int(row[representation]["prompt_tokens"])
                                    * float(
                                        provider.get(
                                            "prompt_token_safety_multiplier", 1.0
                                        )
                                    )
                                )
                                + int(provider["max_output_tokens"])
                            )
                            result = execute_request(
                                provider_name=provider_name,
                                provider=provider,
                                request=request,
                                request_hash=request_sha256(provider_name, provider, request),
                                label_ids=bundle.label_ids,
                            )
                            return {
                                "dataset": dataset,
                                "model_id": model_id,
                                "representation": representation,
                                "repeat": repeat,
                                **result,
                            }

                        completed = 0
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            task_iter = iter(tasks)
                            futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
                            for _ in range(min(workers, len(tasks))):
                                row = next(task_iter)
                                futures[executor.submit(call_one, row)] = row

                            while futures:
                                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                                for future in done:
                                    row = futures.pop(future)
                                    try:
                                        result = future.result()
                                    except Exception as exc:
                                        result = {
                                            "dataset": dataset,
                                            "model_id": row["model_id"],
                                            "representation": representation,
                                            "repeat": repeat,
                                            "status": "api_failed",
                                            "failure_category": "LOCAL_RUNNER_FAILURE",
                                            "error": f"{type(exc).__name__}: {exc}",
                                        }
                                    with write_lock, path.open("a", encoding="utf-8") as handle:
                                        handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                                        handle.flush()
                                    completed += 1
                                    if result.get("status") == "api_failed":
                                        consecutive_api_failures += 1
                                    else:
                                        consecutive_api_failures = 0
                                    print(
                                        f"{provider_name}/{dataset}/{representation}/repeat_{repeat}: "
                                        f"{completed}/{len(tasks)} {result['model_id']} {result['status']}",
                                        flush=True,
                                    )

                                    halt_reason = circuit_breaker_reason(
                                        result,
                                        consecutive_api_failures,
                                        config["protocol"],
                                    )
                                    if halt_reason and not provider_halted:
                                        provider_halted = True
                                        append_run_event(
                                            output_dir,
                                            {
                                                "event": "provider_circuit_open",
                                                "at_utc": utc_now(),
                                                "provider": provider_name,
                                                "configured_model": provider["model"],
                                                "dataset": dataset,
                                                "representation": representation,
                                                "repeat": repeat,
                                                "model_id": result["model_id"],
                                                "failure_category": result.get("failure_category"),
                                                "reason": halt_reason,
                                                "consecutive_api_failures": consecutive_api_failures,
                                                "resume": "fix the cause, rerun preflight, then use --retry-failures",
                                            },
                                        )
                                        print(
                                            f"{provider_name}: circuit opened ({halt_reason}); "
                                            "remaining calls for this provider were not submitted",
                                            flush=True,
                                        )

                                    if not provider_halted:
                                        next_row = next(task_iter, None)
                                        if next_row is not None:
                                            futures[executor.submit(call_one, next_row)] = next_row
            finally:
                bundle.close()


def arrays_for_repeat(
    manifest_rows: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    label_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []
    for row in manifest_rows:
        model_id = str(row["model_id"])
        response = responses[model_id]
        predicted = set(response.get("predicted_label_ids", [])) if response.get("status") == "ok" else set()
        gold = set(row["gold_label_ids"])
        y_true.append([int(label in gold) for label in label_ids])
        y_pred.append([int(label in predicted) for label in label_ids])
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    fields = list(fieldnames) if fieldnames is not None else (list(rows[0]) if rows else [])
    if not fields:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def report(config: dict[str, Any], output_dir: Path, dataset_names: Sequence[str]) -> None:
    manifest, _ = verify_prepared(config, output_dir)
    providers = enabled_providers(config)
    repeats = int(config["protocol"]["repeats"])
    repeat_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for dataset in dataset_names:
        bundle = load_dataset_bundle(dataset, config["datasets"][dataset])
        try:
            dataset_rows = [row for row in manifest if row["dataset"] == dataset]
            for provider_name, provider in providers.items():
                for representation in REPRESENTATIONS:
                    complete_arrays = []
                    metrics_by_repeat = []
                    actual_models = set()
                    for repeat in range(1, repeats + 1):
                        path = response_path(output_dir, provider_name, dataset, representation, repeat)
                        responses = latest_response_map(path)
                        missing = [row["model_id"] for row in dataset_rows if row["model_id"] not in responses]
                        api_failed = [
                            row["model_id"] for row in dataset_rows
                            if responses.get(row["model_id"], {}).get("status") == "api_failed"
                        ]
                        invalid = [
                            row["model_id"] for row in dataset_rows
                            if responses.get(row["model_id"], {}).get("status") == "invalid_response"
                        ]
                        models = {
                            str(responses[row["model_id"]].get("actual_model", ""))
                            for row in dataset_rows
                            if responses.get(row["model_id"], {}).get("status") in {"ok", "invalid_response"}
                        }
                        models.discard("")
                        attestation_failures = 0
                        for manifest_row in dataset_rows:
                            model_id = str(manifest_row["model_id"])
                            response = responses.get(model_id)
                            if not response:
                                continue
                            payload, _ = bundle.input_payload(model_id, representation)
                            prompt = user_prompt(representation, payload, bundle.label_specs)
                            expected_request = build_request(
                                provider,
                                prompt,
                                float(config["protocol"]["temperature"]),
                            )
                            expected_hash = request_sha256(
                                provider_name, provider, expected_request
                            )
                            if response.get("request_sha256") != expected_hash:
                                attestation_failures += 1
                        complete = (
                            not missing
                            and not api_failed
                            and attestation_failures == 0
                            and len(models) == 1
                        )
                        row: dict[str, Any] = {
                            "provider": provider_name,
                            "configured_model": provider["model"],
                            "dataset": dataset,
                            "representation": representation,
                            "repeat": repeat,
                            "status": "COMPLETED" if complete else ("INCOMPLETE" if responses else "NOT_RUN"),
                            "test_examples": len(dataset_rows),
                            "missing": len(missing),
                            "api_failed": len(api_failed),
                            "invalid_responses": len(invalid),
                            "request_attestation_failures": attestation_failures,
                            "format_compliance": (len(dataset_rows) - len(invalid)) / len(dataset_rows) if complete else "",
                            "actual_model": " | ".join(sorted(models)),
                        }
                        if complete:
                            arrays = arrays_for_repeat(dataset_rows, responses, bundle.label_ids)
                            metrics = multilabel_metrics(*arrays)
                            row.update(metrics)
                            complete_arrays.append(arrays)
                            metrics_by_repeat.append(metrics)
                            actual_models.update(models)
                            for item in per_label_metrics(*arrays, bundle.label_ids):
                                spec = bundle.label_specs[bundle.label_ids.index(str(item["label_id"]))]
                                per_label_rows.append(
                                    {
                                        "provider": provider_name,
                                        "dataset": dataset,
                                        "representation": representation,
                                        "repeat": repeat,
                                        "label_id": item["label_id"],
                                        "label_name": spec["name"],
                                        "support": item["support"],
                                        "precision": item["precision"],
                                        "recall": item["recall"],
                                        "f1": item["f1"],
                                    }
                                )
                        for metric in FIVE_METRICS:
                            row.setdefault(metric, "")
                        repeat_rows.append(row)
                        for response in responses.values():
                            if response.get("status") == "api_failed":
                                failure_rows.append(
                                    {
                                        "provider": provider_name,
                                        "dataset": dataset,
                                        "representation": representation,
                                        "repeat": repeat,
                                        "model_id": response.get("model_id", ""),
                                        "failure_category": response.get("failure_category", ""),
                                        "error": response.get("error", ""),
                                    }
                                )

                    summary: dict[str, Any] = {
                        "provider": provider_name,
                        "configured_model": provider["model"],
                        "dataset": dataset,
                        "representation": representation,
                        "status": "COMPLETED" if len(metrics_by_repeat) == repeats else "INCOMPLETE",
                        "completed_repeats": len(metrics_by_repeat),
                        "expected_repeats": repeats,
                        "test_examples": len(dataset_rows),
                        "actual_models": " | ".join(sorted(actual_models)),
                    }
                    if len(metrics_by_repeat) == repeats:
                        intervals = bootstrap_mean_confidence_intervals(
                            complete_arrays,
                            samples=int(config["protocol"]["bootstrap_samples"]),
                            seed=int(config["protocol"]["bootstrap_seed"]),
                        )
                        for metric in FIVE_METRICS:
                            values = [row[metric] for row in metrics_by_repeat]
                            summary[f"{metric}_mean"] = statistics.mean(values)
                            summary[f"{metric}_run_sd"] = (
                                statistics.stdev(values) if len(values) > 1 else ""
                            )
                            summary[f"{metric}_ci_low"] = intervals[metric][0]
                            summary[f"{metric}_ci_high"] = intervals[metric][1]
                    else:
                        for metric in FIVE_METRICS:
                            for suffix in ("mean", "run_sd", "ci_low", "ci_high"):
                                summary[f"{metric}_{suffix}"] = ""
                    summary_rows.append(summary)
        finally:
            bundle.close()

    write_csv(output_dir / "reports" / "repeat_results.csv", repeat_rows)
    write_csv(output_dir / "reports" / "summary.csv", summary_rows)
    write_csv(output_dir / "reports" / "per_label_results.csv", per_label_rows)
    write_csv(
        output_dir / "reports" / "failure_summary.csv",
        failure_rows,
        fieldnames=(
            "provider",
            "dataset",
            "representation",
            "repeat",
            "model_id",
            "failure_category",
            "error",
        ),
    )
    write_json(
        output_dir / "reports" / "report_audit.json",
        {
            "summary_rows": len(summary_rows),
            "complete_summary_rows": sum(row["status"] == "COMPLETED" for row in summary_rows),
            "failures": len(failure_rows),
            "metrics": list(FIVE_METRICS),
            "invalid_response_policy": "score_as_all_negative_and_lower_format_compliance",
            "transport_failure_policy": "cell_incomplete_until_explicit_retry_succeeds",
        },
    )
    print(json.dumps({"summary_rows": len(summary_rows), "failures": len(failure_rows)}, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    output_dir = output_dir_for(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = selected_dataset_names(config, args.datasets)

    if args.mode == "inventory":
        _, audits = build_inventory(config, datasets, output_dir)
        print(json.dumps(audits, ensure_ascii=False, indent=2))
    elif args.mode == "prepare":
        prepare(config, datasets, output_dir)
    elif args.mode == "verify":
        manifest, run_manifest = verify_prepared(config, output_dir)
        print(json.dumps({"status": "PASS", "rows": len(manifest), "run_manifest": run_manifest}, ensure_ascii=False, indent=2))
    elif args.mode == "preflight":
        providers = enabled_providers(config)
        selected = args.providers or list(providers)
        preflight(config, selected, output_dir, args.execute_paid_api_calls)
    elif args.mode == "run":
        run_api(
            config,
            output_dir,
            args.providers,
            datasets,
            args.workers,
            args.retry_failures,
            args.execute_paid_api_calls,
        )
    elif args.mode == "report":
        report(config, output_dir, datasets)


if __name__ == "__main__":
    main()
