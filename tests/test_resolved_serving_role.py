"""Centralized, immutable serving-role resolution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from llmdbenchmark.engine import (
    ResolvedServingRole,
    resolve_engines,
    resolved_serving_role,
    resolved_serving_roles,
    serving_port,
)


def _values() -> dict:
    return {
        "modelservice": {"enabled": True},
        "standalone": {"enabled": False},
        "nok8s": {"enabled": False},
        "engine": {"servicePort": 8000},
        "images": {
            "vllm": {
                "repository": "example/vllm",
                "tag": "latest",
                "pullPolicy": "IfNotPresent",
            }
        },
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "decode": {
            "enabled": True,
            "replicas": 1,
            "engine": {
                "command": (
                    "vllm serve Qwen/Qwen3-0.6B --served-model-name qwen --port 8200"
                )
            },
            "resources": {"limits": {"memory": "16Gi"}},
            "parallelism": {"tensor": 2, "data": 1, "dataLocal": 1},
            "extraEnvVars": [{"name": "CUSTOM", "value": "yes"}],
        },
        "prefill": {"enabled": False, "replicas": 0, "engine": {}},
    }


def test_resolution_publishes_one_complete_role_snapshot() -> None:
    values = _values()
    resolve_engines(values)

    role = resolved_serving_role(values, "decode")
    assert role is not None
    assert role.active is True
    assert role.engine_name == "vllm"
    assert role.launch_mode == "command"
    assert role.model_id == "Qwen/Qwen3-0.6B"
    assert role.served_model_name == "qwen"
    assert role.port == 8200
    assert role.image_repository == "example/vllm"
    assert role.resources["limits"]["memory"] == "16Gi"
    assert role.parallelism["tensor"] == 2
    assert role.environment[0]["name"] == "CUSTOM"

    serialized = values["resolvedServingRoles"]["decode"]
    assert serialized["port"] == 8200
    assert serialized["image"]["repository"] == "example/vllm"
    assert serialized["environment"] == [{"name": "CUSTOM", "value": "yes"}]


def test_resolved_role_is_deeply_immutable() -> None:
    values = _values()
    resolve_engines(values)
    role = resolved_serving_role(values, "decode")
    assert role is not None

    with pytest.raises(FrozenInstanceError):
        role.port = 9000  # type: ignore[misc]
    with pytest.raises(TypeError):
        role.resources["limits"]["memory"] = "1Gi"  # type: ignore[index]
    with pytest.raises(TypeError):
        role.environment[0]["value"] = "no"  # type: ignore[index]


def test_serialized_role_round_trips() -> None:
    values = _values()
    resolve_engines(values)
    serialized = values["resolvedServingRoles"]["decode"]

    restored = ResolvedServingRole.from_dict("decode", serialized)

    assert restored.to_dict() == serialized


def test_published_snapshot_is_authoritative_for_plan_consumers() -> None:
    values = _values()
    resolve_engines(values)

    # Simulate a compatibility field being changed after normalization.  Plan
    # consumers must still agree on the immutable snapshot.
    values["decode"]["engine"]["port"] = 9999
    values["decode"]["engine"]["name"] = "different"

    assert serving_port(values) == 8200
    assert resolved_serving_role(values, "decode").engine_name == "vllm"


def test_legacy_plan_without_snapshots_is_still_readable() -> None:
    values = {
        "standalone": {
            "enabled": True,
            "replicas": 1,
            "engine": {
                "name": "sglang",
                "port": 8200,
                "healthPath": "/health",
                "facts": {"model": "model-a"},
            },
        },
        "engine": {"servicePort": 8000},
        "model": {"name": "model-a"},
    }

    roles = resolved_serving_roles(values)

    assert roles["standalone"].engine_name == "sglang"
    assert roles["standalone"].port == 8200
    assert "resolvedServingRoles" not in values


def test_standalone_records_command_inheritance() -> None:
    values = _values()
    values["modelservice"]["enabled"] = False
    values["standalone"] = {
        "enabled": True,
        "replicas": 1,
        "engine": {},
    }
    resolve_engines(values)

    standalone = resolved_serving_role(values, "standalone")
    decode = resolved_serving_role(values, "decode")
    assert standalone is not None and decode is not None
    assert standalone.active is True
    assert decode.active is False
    assert standalone.inherited_from == "decode"
    assert standalone.command == decode.command


@pytest.mark.parametrize(
    "command,expected",
    [(None, "image-entrypoint"), ("custom-server --port 8000", "command")],
)
def test_command_presence_selects_launch_mode(
    command: str | None, expected: str
) -> None:
    values = _values()
    values["decode"]["engine"] = {
        "name": "custom",
        "command": command,
    }

    resolve_engines(values)

    resolved = values["resolvedServingRoles"]["decode"]
    assert resolved["launchMode"] == expected
