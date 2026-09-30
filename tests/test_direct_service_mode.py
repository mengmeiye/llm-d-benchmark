"""Tests for the gateway.className=none direct modelservice baseline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import yaml

from llmdbenchmark.parser.cluster_resource_resolver import ClusterResourceResolver
from llmdbenchmark.parser.render_plans import RenderPlans
from llmdbenchmark.smoketests.base import BaseSmoketest
from llmdbenchmark.standup.steps.step_07_deploy_router import DeployRouterStep
from llmdbenchmark.utilities.endpoint import (
    find_direct_modelservice_endpoint,
    resolve_direct_service_namespace,
)

_REPO = Path(__file__).resolve().parents[1]

_GPU_SCENARIO = _REPO / "config/scenarios/examples/gpu.yaml"


def _branch(node: dict, key: str) -> dict:
    """setdefault that also replaces an explicit `key:` with no value."""
    child = node.get(key)
    if not isinstance(child, dict):
        child = {}
        node[key] = child
    return child


def _gpu_scenario_in_direct_mode(
    tmp_path: Path,
    *,
    service_port: int = 8000,
    gateway_namespace: str | None = None,
) -> Path:
    """Write a copy of the GPU example that is coherent without a sidecar.

    The shipped scenario runs the llm-d `epponly` topology, where the routing
    sidecar holds ``engine.servicePort`` on each decode pod and decode's command
    binds 8200. Direct mode deletes that sidecar, so the engine has to bind the
    Service port itself -- which means editing the command, the one place a port
    is stated. That edit is the fixture: nothing rewrites the user's command on
    their behalf any more, so a test for direct mode has to make it too.

    The fixture is built by editing the parsed scenario, not its text. A string
    patch silently no-ops when the scenario is re-indented, which leaves the test
    asserting against an unmodified scenario.
    """
    document = yaml.safe_load(_GPU_SCENARIO.read_text(encoding="utf-8"))
    stack = document["scenario"][0]
    common = _branch(stack, "common")
    _branch(common, "engine")["servicePort"] = service_port

    section = _branch(stack, "modelservice")
    gateway = _branch(section, "gateway")
    gateway["className"] = "none"
    if gateway_namespace is not None:
        gateway["namespace"] = gateway_namespace

    decode_engine = _branch(_branch(section, "decode"), "engine")
    command = decode_engine["command"]
    assert "--port 8200" in command, command
    decode_engine["command"] = command.replace("--port 8200", f"--port {service_port}")

    scenario_file = tmp_path / "scenario.yaml"
    scenario_file.write_text(yaml.dump(document, sort_keys=False), encoding="utf-8")
    return scenario_file


def _renderer() -> RenderPlans:
    renderer = RenderPlans.__new__(RenderPlans)
    renderer.logger = MagicMock()
    return renderer


def _passthrough_version_resolver() -> MagicMock:
    """Keep render tests deterministic and independent of image registries."""
    resolver = MagicMock()
    resolver.resolve_all.side_effect = lambda values, **kwargs: values
    return resolver


def test_direct_mode_disables_per_pod_routing_proxy() -> None:
    values = {
        "gateway": {"className": "none"},
        "modelservice": {"enabled": True},
        "routing": {"proxy": {"enabled": True}},
    }

    result = RenderPlans._normalize_direct_service_mode(values)

    assert result["routing"]["proxy"]["enabled"] is False


def test_direct_mode_leaves_the_decode_command_alone() -> None:
    """Direct mode must not rewrite the port inside the user's command.

    It used to: the renderer swapped the decode command's port for the Service
    port so the baseline worked without the sidecar. That silently made the
    running engine disagree with the command the user wrote and read back in
    the plan, and it only ever worked because the port was an env var
    llm-d-benchmark itself defined. A pasted `--port 8200` would have been
    rewritten too, or (with any other spelling) missed entirely.

    The command is now the authority. Direct mode turns the sidecar off and
    nothing else; when the command binds the wrong port,
    ``_validate_engine_ports`` says so instead of papering over it.
    """
    command = "vllm serve model --port 8200"
    values = {
        "gateway": {"className": "none"},
        "modelservice": {"enabled": True},
        "engine": {"servicePort": 8000},
        "decode": {"replicas": 1, "engine": {"command": command, "port": 8200}},
    }

    result = RenderPlans._normalize_direct_service_mode(values)

    assert result["decode"]["engine"]["command"] == command
    assert result["routing"]["proxy"]["enabled"] is False


def test_direct_mode_reports_a_decode_command_that_misses_the_service_port() -> None:
    """The error has to name the port to change, and where to change it.

    Without the sidecar nothing bridges 8000 to 8200, so this stack comes up
    green and answers nothing -- the failure mode that makes this an error
    rather than a warning.
    """
    values = {
        "gateway": {"className": "none"},
        "modelservice": {"enabled": True},
        "engine": {"servicePort": 8000},
        "decode": {"replicas": 1, "engine": {"command": "...", "port": 8200}},
    }
    RenderPlans._normalize_direct_service_mode(values)

    errors = RenderPlans._validate_engine_ports(values, "stack")

    assert len(errors) == 1, errors
    assert "routing sidecar is disabled" in errors[0]
    assert "decode.engine.command" in errors[0]
    assert "8000" in errors[0]


def test_direct_mode_accepts_a_decode_command_that_binds_the_service_port() -> None:
    values = {
        "gateway": {"className": "none"},
        "modelservice": {"enabled": True},
        "engine": {"servicePort": 8000},
        "decode": {
            "replicas": 1,
            "engine": {"command": "vllm serve model --port 8000", "port": 8000},
        },
    }
    RenderPlans._normalize_direct_service_mode(values)

    assert RenderPlans._validate_engine_ports(values, "stack") == []


def test_direct_mode_rejects_prefill_routing() -> None:
    values = {
        "gateway": {"className": "none"},
        "modelservice": {"enabled": True},
        "decode": {"replicas": 1},
        "prefill": {"enabled": True, "replicas": 1},
    }

    errors = RenderPlans._validate_direct_service_constraints(values, "stack")

    assert any("bypasses P/D routing" in error for error in errors)


def test_direct_mode_allows_multiple_independent_stacks() -> None:
    values = {
        "gateway": {"className": "none"},
        "modelservice": {"enabled": True},
        "decode": {"replicas": 1},
        "prefill": {"enabled": False, "replicas": 0},
    }

    assert RenderPlans._validate_direct_service_constraints(values, "stack") == []


def test_none_is_supported_modelservice_gateway_class() -> None:
    renderer = _renderer()
    renderer.cli_gateway_class = "none"
    values = {
        "gateway": {"className": "epponly"},
        "modelservice": {"enabled": True},
    }

    result = renderer._resolve_gateway_class(values)

    assert result["gateway"]["className"] == "none"


def test_direct_endpoint_prefers_named_http_port() -> None:
    service = {
        "spec": {
            "clusterIP": "10.0.0.42",
            "ports": [
                {"name": "metrics", "port": 9090},
                {"name": "http", "port": 8000},
            ],
        }
    }
    cmd = MagicMock()
    cmd.kube.return_value = SimpleNamespace(
        success=True,
        stdout=json.dumps(service),
    )

    endpoint = find_direct_modelservice_endpoint(cmd, "bench", "model-id")

    assert endpoint == ("10.0.0.42", "model-id-direct", "8000")
    cmd.kube.assert_called_once_with(
        "get",
        "service",
        "model-id-direct",
        "--namespace",
        "bench",
        "-o",
        "json",
        check=False,
    )


def test_direct_service_namespace_prefers_explicit_gateway_namespace() -> None:
    plan_config = {"gateway": {"namespace": "model-serving"}}

    assert resolve_direct_service_namespace(plan_config, "benchmark") == "model-serving"
    assert resolve_direct_service_namespace({}, "benchmark") == "benchmark"
    assert (
        resolve_direct_service_namespace(
            {"gateway": {"namespace": "auto"}}, "benchmark"
        )
        == "benchmark"
    )


def test_direct_smoketest_discovers_service_in_gateway_namespace() -> None:
    service = {"spec": {"clusterIP": "10.0.0.42", "ports": [{"port": 8000}]}}
    cmd = MagicMock()
    cmd.kube.return_value = SimpleNamespace(
        success=True,
        stdout=json.dumps(service),
    )
    context = MagicMock()
    context.deployed_methods = ["modelservice"]
    context.require_namespace.return_value = "benchmark"
    context.dry_run = False
    plan_config = {
        "gateway": {"className": "none", "namespace": "model-serving"},
        "model_id_label": "model-id",
        "routing": {"servicePort": 8000},
    }

    endpoint = BaseSmoketest.discover_endpoint(cmd, context, plan_config)

    assert endpoint == ("10.0.0.42", "8000", False)
    cmd.kube.assert_called_once_with(
        "get",
        "service",
        "model-id-direct",
        "--namespace",
        "model-serving",
        "-o",
        "json",
        check=False,
    )


def test_direct_mode_skips_router_deployment(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"className": "none"}}),
        encoding="utf-8",
    )
    cmd = MagicMock()
    context = MagicMock()
    context.require_cmd.return_value = cmd

    result = DeployRouterStep().execute(context, tmp_path)

    assert result.success
    assert result.message == "Direct Service mode does not deploy a router"
    cmd.helmfile.assert_not_called()
    cmd.wait_for_pods.assert_not_called()


def test_gpu_example_renders_plain_service_without_router(tmp_path: Path) -> None:
    """Direct mode: one Helm release, a plain Service, no Gateway or EPP."""
    scenario_file = _gpu_scenario_in_direct_mode(tmp_path)
    output_dir = tmp_path / "rendered"
    logger = MagicMock()
    result = RenderPlans(
        template_dir=_REPO / "config/templates/jinja",
        defaults_file=_REPO / "config/templates/values/defaults.yaml",
        scenarios_file=scenario_file,
        output_dir=output_dir,
        logger=logger,
        version_resolver=_passthrough_version_resolver(),
        cluster_resource_resolver=ClusterResourceResolver(logger=logger, dry_run=True),
        cli_gateway_class="none",
    ).eval()

    assert not result.has_errors, result.to_dict()
    stack_dirs = [path.parent for path in output_dir.rglob("config.yaml")]
    assert len(stack_dirs) == 1
    stack_dir = stack_dirs[0]

    config = yaml.safe_load((stack_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["gateway"]["className"] == "none"
    assert config["routing"]["proxy"]["enabled"] is False

    helmfile = yaml.safe_load_all(
        (stack_dir / "10_helmfile-main.yaml").read_text(encoding="utf-8")
    )
    releases = [
        release
        for document in helmfile
        if document
        for release in document.get("releases", [])
    ]
    assert [release["name"] for release in releases] == [
        f"{config['model_id_label']}-ms"
    ]

    direct_service = yaml.safe_load(
        (stack_dir / "13a_modelservice-direct-service.yaml").read_text(encoding="utf-8")
    )
    assert direct_service["kind"] == "Service"
    assert direct_service["metadata"]["name"] == f"{config['model_id_label']}-direct"
    assert direct_service["metadata"]["namespace"] == config["gateway"]["namespace"]
    assert direct_service["spec"]["selector"] == {
        "llm-d.ai/model": config["model_id_label"],
        "llm-d.ai/role": "decode",
    }
    assert direct_service["spec"]["ports"][0]["targetPort"] == 8000

    assert not (stack_dir / "08_httproute.yaml").read_text(encoding="utf-8").strip()
    assert not (stack_dir / "11_infra.yaml").read_text(encoding="utf-8").strip()

    modelservice_values = yaml.safe_load(
        (stack_dir / "13_ms-values.yaml").read_text(encoding="utf-8")
    )
    assert modelservice_values["routing"]["proxy"]["enabled"] is False

    # The command reaches the container verbatim: the port the scenario wrote is
    # the port the engine binds, and it is the Service's target. Nothing
    # substitutes a benchmark-owned variable in for it.
    command = modelservice_values["decode"]["containers"][0]["args"][0]
    assert "--port 8000" in command
    assert "VLLM_" not in command


def test_direct_service_uses_gateway_namespace_and_decode_target_port(
    tmp_path: Path,
) -> None:
    """The Service lands beside the model and targets the engine's own port.

    ``routing.servicePort`` is what clients dial; ``engine.servicePort`` is what
    the pod listens on, which in direct mode is the port decode's command binds.
    They are allowed to differ, so the template has to read each from its own
    place rather than assume one number.
    """
    scenario_file = _gpu_scenario_in_direct_mode(
        tmp_path,
        service_port=8100,
        gateway_namespace="model-serving",
    )
    document = yaml.safe_load(scenario_file.read_text(encoding="utf-8"))
    _branch(document["scenario"][0]["modelservice"], "routing")["servicePort"] = 8000
    scenario_file.write_text(yaml.dump(document, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "rendered"
    logger = MagicMock()

    result = RenderPlans(
        template_dir=_REPO / "config/templates/jinja",
        defaults_file=_REPO / "config/templates/values/defaults.yaml",
        scenarios_file=scenario_file,
        output_dir=output_dir,
        logger=logger,
        version_resolver=_passthrough_version_resolver(),
        cluster_resource_resolver=ClusterResourceResolver(logger=logger, dry_run=True),
    ).eval()

    assert not result.has_errors, result.to_dict()
    stack_dir = next(path.parent for path in output_dir.rglob("config.yaml"))
    config = yaml.safe_load((stack_dir / "config.yaml").read_text(encoding="utf-8"))
    direct_service = yaml.safe_load(
        (stack_dir / "13a_modelservice-direct-service.yaml").read_text(encoding="utf-8")
    )

    assert config["namespace"]["name"] != "model-serving"
    assert config["gateway"]["namespace"] == "model-serving"
    assert direct_service["metadata"]["namespace"] == "model-serving"
    assert direct_service["spec"]["ports"][0]["port"] == 8000
    assert direct_service["spec"]["ports"][0]["targetPort"] == 8100
