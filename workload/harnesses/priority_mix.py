#!/usr/bin/env python3

"""Priority-mix workload harness.

This harness sends OpenAI-compatible requests with different
``x-llm-d-inference-objective`` values so EPP priority bands can be tested with
mixed traffic classes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

OBJECTIVE_HEADER = "x-llm-d-inference-objective"
FAIRNESS_HEADER = "x-llm-d-inference-fairness-id"


@dataclass(frozen=True)
class TrafficClass:
    name: str
    weight: float
    headers: dict[str, str] = field(default_factory=dict)
    priority: int | None = None
    request: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestResult:
    traffic_class: str
    status_code: int
    latency_ms: float
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    output_tokens: int = 0
    error: str | None = None
    request_index: int = 0
    class_request_index: int = 0
    priority: int | None = None
    prefix_group: int | None = None


def setup_logger(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("priority_mix")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (
        logging.FileHandler(results_dir / "stdout.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_profile(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as profile_file:
        profile = yaml.safe_load(profile_file) or {}
    if not isinstance(profile, dict):
        raise ValueError(f"profile must be a YAML mapping: {path}")
    return profile


def traffic_classes(profile: dict[str, Any]) -> list[TrafficClass]:
    classes = profile.get("trafficClasses") or profile.get("traffic_classes")
    if not classes:
        raise ValueError("profile must define trafficClasses")
    parsed: list[TrafficClass] = []
    for item in classes:
        if not isinstance(item, dict):
            raise ValueError("each traffic class must be a mapping")
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError("traffic class name is required")
        headers = {
            str(key): str(value) for key, value in (item.get("headers") or {}).items()
        }
        objective = item.get("objective")
        if objective:
            headers.setdefault(OBJECTIVE_HEADER, str(objective))
        fairness_id = item.get("fairnessID") or item.get("fairness_id")
        if fairness_id:
            headers.setdefault(FAIRNESS_HEADER, str(fairness_id))
        request_overrides = dict(item.get("request") or {})
        for key in (
            "prompt",
            "promptTemplate",
            "prompt_template",
            "promptRepeat",
            "max_tokens",
        ):
            if key in item:
                request_overrides[key] = item[key]
        priority = item.get("priority")
        parsed.append(
            TrafficClass(
                name=name,
                weight=float(item.get("weight", 1)),
                headers=headers,
                priority=int(priority) if priority is not None else None,
                request=request_overrides,
            )
        )
    if any(item.weight <= 0 for item in parsed):
        raise ValueError("traffic class weights must be positive")
    names = [item.name for item in parsed]
    if len(set(names)) != len(names):
        raise ValueError("traffic class names must be unique")
    return parsed


def weighted_schedule(
    classes: list[TrafficClass], total_requests: int
) -> list[TrafficClass]:
    total_weight = sum(item.weight for item in classes)
    current = [0.0 for _ in classes]
    schedule: list[TrafficClass] = []
    for _ in range(total_requests):
        for index, item in enumerate(classes):
            current[index] += item.weight
        selected_index = max(range(len(classes)), key=lambda index: current[index])
        current[selected_index] -= total_weight
        schedule.append(classes[selected_index])
    return schedule


def build_payload(
    profile: dict[str, Any],
    traffic_class: TrafficClass | None = None,
    request_index: int = 0,
) -> dict[str, Any]:
    model = str(
        profile.get("model") or os.environ.get("LLMDBENCH_DEPLOY_CURRENT_MODEL", "")
    )
    if not model:
        raise ValueError(
            "model is required in profile or LLMDBENCH_DEPLOY_CURRENT_MODEL"
        )

    request = dict(profile.get("request") or {})
    if traffic_class is not None:
        request.update(traffic_class.request)
    api = profile.get("api") or {}
    prompt = render_prompt(request, traffic_class, request_index)
    max_tokens = int(request.get("max_tokens", 32))

    if str(api.get("type", "chat")).lower() in {"completion", "completions"}:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": bool(request.get("stream", False)),
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": bool(request.get("stream", False)),
        }

    if payload["stream"] and request.get("includeUsage", False):
        payload["stream_options"] = {"include_usage": True}

    # Most llm-d priority experiments use InferenceObjective headers and let
    # the router choose a backend priority. Engine-local policies sometimes
    # need the workload value in the OpenAI request body instead. Keep that
    # behavior opt-in so existing router-priority profiles do not change.
    if (
        profile.get("sendPriorityInBody", False)
        and traffic_class is not None
        and traffic_class.priority is not None
    ):
        payload["priority"] = int(traffic_class.priority)

    return payload


def render_prompt(
    request: dict[str, Any],
    traffic_class: TrafficClass | None,
    request_index: int,
) -> str:
    prefix_groups = max(1, int(request.get("prefixGroups", 1)))
    format_values = {
        "traffic_class": traffic_class.name if traffic_class else "default",
        "request_index": request_index,
        "variation": request_index,
        "group": request_index % prefix_groups,
    }
    shared_template = request.get("sharedPromptTemplate") or request.get(
        "shared_prompt_template"
    )
    shared_prompt = ""
    if shared_template:
        shared_prompt = str(shared_template).format(**format_values) * int(
            request.get("sharedPromptRepeat", 1)
        )

    template = request.get("promptTemplate") or request.get("prompt_template")
    prompt = str(
        template or request.get("prompt", "Say hello from the priority mix benchmark.")
    )
    if template:
        prompt = prompt.format(**format_values)
    return shared_prompt + prompt * int(request.get("promptRepeat", 1))


def prefix_group(
    profile: dict[str, Any], traffic_class: TrafficClass, class_request_index: int
) -> int:
    """Return the reusable-prefix group selected within one traffic class."""
    request = dict(profile.get("request") or {})
    request.update(traffic_class.request)
    prefix_groups = max(1, int(request.get("prefixGroups", 1)))
    return class_request_index % prefix_groups


def request_url(profile: dict[str, Any]) -> str:
    endpoint = str(
        profile.get("endpoint_url")
        or os.environ.get("LLMDBENCH_HARNESS_STACK_ENDPOINT_URL", "")
    ).rstrip("/")
    if not endpoint:
        raise ValueError(
            "endpoint_url is required in profile or LLMDBENCH_HARNESS_STACK_ENDPOINT_URL"
        )
    api = profile.get("api") or {}
    path = str(api.get("path") or "/v1/chat/completions")
    if not path.startswith("/"):
        path = "/" + path
    return endpoint + path


def send_request(
    url: str,
    payload: dict[str, Any],
    traffic_class: TrafficClass,
    timeout_seconds: float,
    *,
    request_index: int = 0,
    class_request_index: int = 0,
    prefix_group_id: int | None = None,
) -> RequestResult:
    headers = {"Content-Type": "application/json", **traffic_class.headers}
    start = time.perf_counter()
    try:
        with requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
            stream=bool(payload.get("stream")),
        ) as response:
            if payload.get("stream"):
                return read_streaming_response(
                    response,
                    traffic_class,
                    start,
                    request_index=request_index,
                    class_request_index=class_request_index,
                    prefix_group_id=prefix_group_id,
                )
            latency_ms = (time.perf_counter() - start) * 1000
            output_tokens = 0
            try:
                output_tokens = int(
                    (response.json().get("usage") or {}).get("completion_tokens", 0)
                )
            except (requests.JSONDecodeError, TypeError, ValueError):
                pass
            return RequestResult(
                traffic_class=traffic_class.name,
                status_code=response.status_code,
                latency_ms=latency_ms,
                output_tokens=output_tokens,
                request_index=request_index,
                class_request_index=class_request_index,
                priority=traffic_class.priority,
                prefix_group=prefix_group_id,
            )
    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            traffic_class=traffic_class.name,
            status_code=0,
            latency_ms=latency_ms,
            error=str(exc),
            request_index=request_index,
            class_request_index=class_request_index,
            priority=traffic_class.priority,
            prefix_group=prefix_group_id,
        )


def read_streaming_response(
    response: requests.Response,
    traffic_class: TrafficClass,
    start: float,
    *,
    request_index: int = 0,
    class_request_index: int = 0,
    prefix_group_id: int | None = None,
) -> RequestResult:
    first_token_time: float | None = None
    last_token_time: float | None = None
    output_tokens = 0
    reported_output_tokens: int | None = None
    error: str | None = None
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = str(raw_line)
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                reported_output_tokens = int(usage["completion_tokens"])
            if streaming_chunk_has_content(chunk):
                now = time.perf_counter()
                first_token_time = first_token_time or now
                last_token_time = now
                output_tokens += 1
    except requests.RequestException as exc:
        error = str(exc)
    latency_ms = (time.perf_counter() - start) * 1000
    ttft_ms = ((first_token_time - start) * 1000) if first_token_time else None
    tpot_ms = None
    if first_token_time and last_token_time and output_tokens > 1:
        tpot_ms = ((last_token_time - first_token_time) * 1000) / (output_tokens - 1)
    return RequestResult(
        traffic_class=traffic_class.name,
        status_code=response.status_code,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        output_tokens=(
            reported_output_tokens
            if reported_output_tokens is not None
            else output_tokens
        ),
        error=error,
        request_index=request_index,
        class_request_index=class_request_index,
        priority=traffic_class.priority,
        prefix_group=prefix_group_id,
    )


def streaming_chunk_has_content(data: str | dict[str, Any]) -> bool:
    if isinstance(data, str):
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return False
    else:
        chunk = data
    for choice in chunk.get("choices", []):
        delta = choice.get("delta") or {}
        if delta.get("content"):
            return True
        if choice.get("text"):
            return True
    return False


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(
    results: list[RequestResult],
    classes: list[TrafficClass],
    elapsed_seconds: float = 0,
) -> dict[str, Any]:
    def successful(result: RequestResult) -> bool:
        return result.error is None and 200 <= result.status_code < 300

    by_class: dict[str, list[RequestResult]] = {item.name: [] for item in classes}
    for result in results:
        by_class.setdefault(result.traffic_class, []).append(result)

    class_summaries: dict[str, Any] = {}
    for item in classes:
        class_results = by_class.get(item.name, [])
        successful_results = [result for result in class_results if successful(result)]
        latencies = [result.latency_ms for result in successful_results]
        ttfts = [
            result.ttft_ms
            for result in successful_results
            if result.ttft_ms is not None
        ]
        tpots = [
            result.tpot_ms
            for result in successful_results
            if result.tpot_ms is not None
        ]
        successes = sum(1 for result in class_results if successful(result))
        errors = [
            result
            for result in class_results
            if result.error or result.status_code >= 400
        ]
        class_summaries[item.name] = {
            "configured_weight": item.weight,
            "configured_priority": item.priority,
            "headers": item.headers,
            "requests": len(class_results),
            "successes": successes,
            "errors": len(errors),
            "latency_ms": {
                "avg": sum(latencies) / len(latencies) if latencies else 0.0,
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
            },
            "ttft_ms": {
                "avg": sum(ttfts) / len(ttfts) if ttfts else 0.0,
                "p50": percentile(ttfts, 0.50),
                "p95": percentile(ttfts, 0.95),
                "p99": percentile(ttfts, 0.99),
            },
            "tpot_ms": {
                "avg": sum(tpots) / len(tpots) if tpots else 0.0,
                "p50": percentile(tpots, 0.50),
                "p95": percentile(tpots, 0.95),
                "p99": percentile(tpots, 0.99),
            },
            "output_tokens": sum(result.output_tokens for result in successful_results),
            "request_throughput_per_second": (
                successes / elapsed_seconds if elapsed_seconds > 0 else 0.0
            ),
            "output_token_throughput_per_second": (
                sum(result.output_tokens for result in successful_results)
                / elapsed_seconds
                if elapsed_seconds > 0
                else 0.0
            ),
        }

    successes = sum(1 for result in results if successful(result))
    output_tokens = sum(
        result.output_tokens for result in results if successful(result)
    )
    return {
        "total_requests": len(results),
        "successes": successes,
        "errors": sum(
            1 for result in results if result.error or result.status_code >= 400
        ),
        "elapsed_seconds": elapsed_seconds,
        "request_throughput_per_second": (
            successes / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "output_token_throughput_per_second": (
            output_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "traffic_classes": class_summaries,
    }


def run(profile: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    load = profile.get("load") or {}
    duration_seconds = float(load.get("duration_seconds", 30))
    rate_per_second = float(load.get("rate_per_second", 1))
    max_in_flight = int(load.get("max_in_flight", max(1, math.ceil(rate_per_second))))
    timeout_seconds = float(load.get("request_timeout_seconds", 60))
    total_requests = int(
        load.get("total_requests", max(1, duration_seconds * rate_per_second))
    )

    classes = traffic_classes(profile)
    schedule = weighted_schedule(classes, total_requests)
    url = request_url(profile)

    logger.info(
        "running priority-mix workload url=%s total_requests=%d", url, total_requests
    )
    logger.info("traffic classes: %s", ", ".join(item.name for item in classes))

    results: list[RequestResult] = []
    class_request_counts: dict[str, int] = {item.name: 0 for item in classes}
    interval = 1 / rate_per_second if rate_per_second > 0 else 0
    run_started = time.perf_counter()
    next_send = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_in_flight) as executor:
        futures = []
        for index in range(total_requests):
            delay = next_send - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            traffic_class = schedule[index]
            class_request_index = class_request_counts[traffic_class.name]
            class_request_counts[traffic_class.name] += 1
            payload = build_payload(profile, traffic_class, class_request_index)
            futures.append(
                executor.submit(
                    send_request,
                    url,
                    payload,
                    traffic_class,
                    timeout_seconds,
                    request_index=index,
                    class_request_index=class_request_index,
                    prefix_group_id=prefix_group(
                        profile, traffic_class, class_request_index
                    ),
                )
            )
            next_send += interval
        for future in as_completed(futures):
            results.append(future.result())

    elapsed_seconds = time.perf_counter() - run_started
    results.sort(key=lambda result: result.request_index)
    return {
        "summary": summarize(results, classes, elapsed_seconds),
        "requests": [result.__dict__ for result in results],
    }


def write_run_metadata(
    results_dir: Path, start: datetime, stop: datetime, rc: int
) -> None:
    metadata = {
        "harness_start": start.isoformat(),
        "harness_stop": stop.isoformat(),
        "harness_delta": f"PT{(stop - start).total_seconds()}S",
        "harness_args": f"--workload {os.environ.get('LLMDBENCH_RUN_EXPERIMENT_HARNESS_WORKLOAD_NAME', '')}",
        "harness_version": "unknown",
        "harness_name": "priority-mix",
        "harness_workload": os.environ.get(
            "LLMDBENCH_RUN_EXPERIMENT_HARNESS_WORKLOAD_NAME", ""
        ),
        "harness_rc": str(rc),
        "experiment_id": os.environ.get("LLMDBENCH_RUN_EXPERIMENT_ID", ""),
        "model": os.environ.get("LLMDBENCH_DEPLOY_CURRENT_MODEL", ""),
        "endpoint_url": os.environ.get("LLMDBENCH_HARNESS_STACK_ENDPOINT_URL", ""),
        "namespace": os.environ.get("LLMDBENCH_VLLM_COMMON_NAMESPACE", ""),
        "description_text": os.environ.get("LLMDBENCH_DESCRIPTION_TEXT", ""),
        "description_keywords": os.environ.get("LLMDBENCH_DESCRIPTION_KEYWORDS", ""),
    }
    with (results_dir / "run_metadata.yaml").open(
        "w", encoding="utf-8"
    ) as metadata_file:
        yaml.safe_dump(metadata, metadata_file, sort_keys=False)


def main() -> int:
    results_dir = Path(os.environ["LLMDBENCH_RUN_EXPERIMENT_RESULTS_DIR"])
    workspace_dir = Path(os.environ.get("LLMDBENCH_RUN_WORKSPACE_DIR", "/workspace"))
    workload_name = os.environ.get(
        "LLMDBENCH_RUN_EXPERIMENT_HARNESS_WORKLOAD_NAME", "priority_mix.yaml"
    )
    profile_path = workspace_dir / "profiles" / "priority-mix" / workload_name
    logger = setup_logger(results_dir)

    start = datetime.now(timezone.utc)
    rc = 0
    try:
        profile = load_profile(profile_path)
        output = run(profile, logger)
        results_file = results_dir / "results.json"
        with results_file.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, sort_keys=True)
        logger.info("results written to %s", results_file)
        if (profile.get("load") or {}).get("fail_on_error", False):
            rc = 1 if output["summary"]["errors"] else 0
    except Exception:  # pylint: disable=broad-exception-caught
        rc = 1
        logger.exception("priority-mix workload failed")
    finally:
        stop = datetime.now(timezone.utc)
        write_run_metadata(results_dir, start, stop, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
