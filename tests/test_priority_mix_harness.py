"""Tests for the priority-mix harness."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_HARNESS_PATH = (
    Path(__file__).resolve().parent.parent
    / "workload"
    / "harnesses"
    / "priority_mix.py"
)
_spec = importlib.util.spec_from_file_location("priority_mix", _HARNESS_PATH)
priority_mix = importlib.util.module_from_spec(_spec)
sys.modules["priority_mix"] = priority_mix
_spec.loader.exec_module(priority_mix)


def test_traffic_classes_add_objective_and_fairness_headers() -> None:
    classes = priority_mix.traffic_classes(
        {
            "trafficClasses": [
                {
                    "name": "critical",
                    "weight": 1,
                    "objective": "app-critical",
                    "fairnessID": "tenant-a",
                    "priority": 100,
                }
            ]
        }
    )

    assert classes[0].headers == {
        priority_mix.OBJECTIVE_HEADER: "app-critical",
        priority_mix.FAIRNESS_HEADER: "tenant-a",
    }
    assert classes[0].priority == 100


def test_traffic_class_request_overrides_prompt_template() -> None:
    profile = {
        "model": "test-model",
        "request": {"prompt": "global", "max_tokens": 8},
        "trafficClasses": [
            {
                "name": "critical",
                "weight": 1,
                "objective": "app-critical",
                "request": {
                    "promptTemplate": "critical prefix {traffic_class} {variation}. ",
                    "promptRepeat": 2,
                },
            }
        ],
    }

    traffic_class = priority_mix.traffic_classes(profile)[0]
    payload = priority_mix.build_payload(profile, traffic_class, 7)

    assert payload["messages"][0]["content"] == (
        "critical prefix critical 7. critical prefix critical 7. "
    )
    assert payload["max_tokens"] == 8


def test_build_payload_can_send_vllm_priority_in_body() -> None:
    profile = {
        "model": "test-model",
        "sendPriorityInBody": True,
        "api": {"type": "completions"},
        "trafficClasses": [
            {
                "name": "fast-tier",
                "weight": 1,
                "priority": 0,
            }
        ],
    }

    traffic_class = priority_mix.traffic_classes(profile)[0]
    payload = priority_mix.build_payload(profile, traffic_class)

    assert payload["priority"] == 0


def test_build_payload_requests_streaming_usage() -> None:
    profile = {
        "model": "test-model",
        "request": {"stream": True, "includeUsage": True},
        "trafficClasses": [{"name": "fast-tier", "weight": 1}],
    }

    traffic_class = priority_mix.traffic_classes(profile)[0]
    payload = priority_mix.build_payload(profile, traffic_class)

    assert payload["stream_options"] == {"include_usage": True}


def test_build_payload_does_not_send_priority_by_default() -> None:
    profile = {
        "model": "test-model",
        "trafficClasses": [
            {
                "name": "router-managed",
                "weight": 1,
                "priority": 100,
            }
        ],
    }

    traffic_class = priority_mix.traffic_classes(profile)[0]
    payload = priority_mix.build_payload(profile, traffic_class)

    assert "priority" not in payload


def test_render_prompt_builds_reusable_prefix_groups() -> None:
    request = {
        "prefixGroups": 2,
        "sharedPromptTemplate": "group-{group} common ",
        "sharedPromptRepeat": 2,
        "promptTemplate": "request-{request_index}",
    }

    first = priority_mix.render_prompt(request, None, 0)
    reused = priority_mix.render_prompt(request, None, 2)
    other = priority_mix.render_prompt(request, None, 1)

    assert first == "group-0 common group-0 common request-0"
    assert reused == "group-0 common group-0 common request-2"
    assert other.startswith("group-1 common group-1 common ")


def test_run_uses_class_local_prefix_groups_and_records_request_identity(
    monkeypatch,
) -> None:
    def fake_send_request(
        _url,
        _payload,
        traffic_class,
        _timeout_seconds,
        **metadata,
    ):
        return priority_mix.RequestResult(
            traffic_class=traffic_class.name,
            status_code=200,
            latency_ms=1,
            output_tokens=2,
            request_index=metadata["request_index"],
            class_request_index=metadata["class_request_index"],
            priority=traffic_class.priority,
            prefix_group=metadata["prefix_group_id"],
        )

    monkeypatch.setattr(priority_mix, "send_request", fake_send_request)
    profile = {
        "endpoint_url": "http://unused.invalid",
        "model": "test-model",
        "load": {"rate_per_second": 0, "max_in_flight": 2, "total_requests": 8},
        "trafficClasses": [
            {
                "name": "fast-tier",
                "weight": 1,
                "priority": 0,
                "request": {"prefixGroups": 2},
            },
            {
                "name": "slow-tier",
                "weight": 1,
                "priority": 5,
                "request": {"prefixGroups": 2},
            },
        ],
    }

    output = priority_mix.run(profile, priority_mix.logging.getLogger("test"))

    assert [row["request_index"] for row in output["requests"]] == list(range(8))
    for class_name, priority in (("fast-tier", 0), ("slow-tier", 5)):
        rows = [row for row in output["requests"] if row["traffic_class"] == class_name]
        assert [row["class_request_index"] for row in rows] == [0, 1, 2, 3]
        assert [row["prefix_group"] for row in rows] == [0, 1, 0, 1]
        assert {row["priority"] for row in rows} == {priority}

    assert output["summary"]["request_throughput_per_second"] > 0
    assert output["summary"]["output_token_throughput_per_second"] > 0


def test_priority_mix_sends_each_objective_header() -> None:
    seen: list[str] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            with lock:
                seen.append(self.headers.get(priority_mix.OBJECTIVE_HEADER, ""))
            body = json.dumps({"id": "ok", "choices": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        profile = {
            "endpoint_url": f"http://127.0.0.1:{server.server_port}",
            "model": "test-model",
            "load": {
                "duration_seconds": 1,
                "rate_per_second": 100,
                "max_in_flight": 4,
                "total_requests": 4,
            },
            "trafficClasses": [
                {"name": "critical", "weight": 1, "objective": "app-critical"},
                {"name": "normal", "weight": 1, "objective": "app-normal"},
            ],
        }
        output = priority_mix.run(profile, priority_mix.logging.getLogger("test"))
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert output["summary"]["total_requests"] == 4
    assert output["summary"]["errors"] == 0
    assert seen.count("app-critical") == 2
    assert seen.count("app-normal") == 2


def test_send_request_sets_error_not_ttft_on_connection_failure() -> None:
    traffic_class = priority_mix.TrafficClass(name="critical", weight=1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        unused_port = probe.getsockname()[1]

    result = priority_mix.send_request(
        f"http://127.0.0.1:{unused_port}/v1/chat/completions",
        {"model": "test-model", "messages": [], "stream": False},
        traffic_class,
        timeout_seconds=2,
    )

    assert result.status_code == 0
    assert result.error is not None
    assert result.ttft_ms is None

    summary = priority_mix.summarize([result], [traffic_class])
    assert summary["successes"] == 0
    assert summary["errors"] == 1


def test_priority_mix_reports_streaming_ttft_and_tpot() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for token in ("hello", " ", "world"):
                time.sleep(0.01)
                chunk = {"choices": [{"delta": {"content": token}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, *_: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        profile = {
            "endpoint_url": f"http://127.0.0.1:{server.server_port}",
            "model": "test-model",
            "load": {
                "duration_seconds": 1,
                "rate_per_second": 100,
                "max_in_flight": 1,
                "total_requests": 1,
            },
            "request": {"stream": True},
            "trafficClasses": [
                {"name": "critical", "weight": 1, "objective": "app-critical"},
            ],
        }
        output = priority_mix.run(profile, priority_mix.logging.getLogger("test"))
    finally:
        server.shutdown()
        thread.join(timeout=5)

    summary = output["summary"]["traffic_classes"]["critical"]
    assert summary["requests"] == 1
    assert summary["ttft_ms"]["avg"] > 0
    assert summary["tpot_ms"]["avg"] > 0
    assert summary["output_tokens"] == 3
