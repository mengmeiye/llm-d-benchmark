"""Serving-port selection across deployment methods."""

from __future__ import annotations

from llmdbenchmark.engine import serving_port

SGLANG_8200 = {
    "name": "sglang",
    "port": 8200,
    "healthPath": "/health",
    "command": "sglang serve m --port 8200",
}


def test_standalone_port_comes_from_the_command_not_the_service():
    values = {
        "standalone": {"engine": SGLANG_8200},
        "engine": {"servicePort": 8000},
    }
    assert serving_port(values) == 8200


def test_decode_port_wins_for_a_modelservice_stack():
    values = {"decode": {"engine": SGLANG_8200}, "engine": {"servicePort": 8000}}
    assert serving_port(values) == 8200


def test_falls_back_to_the_service_port_with_no_resolved_engine():
    """A bare values tree (`--dry-run`, an unresolved plan) still answers."""
    assert serving_port({"engine": {"servicePort": 8001}}) == 8001


def test_falls_back_past_an_unusable_port():
    values = {
        "standalone": {"engine": {"name": "x", "port": "", "command": "x serve m"}},
        "engine": {"servicePort": 8080},
    }
    assert serving_port(values) == 8080


def test_answers_with_nothing_to_go_on():
    assert serving_port({}) == 8000
