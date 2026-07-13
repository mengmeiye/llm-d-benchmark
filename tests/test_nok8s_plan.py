"""Tests for the nok8s (no-Kubernetes) deployment method.

Covers render-time gating (templates + config.yaml flags) and the step
should_skip selection logic, without needing a cluster or GPUs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llmdbenchmark.parser.render_plans import RenderPlans

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "config" / "templates" / "jinja"
DEFAULTS_FILE = REPO_ROOT / "config" / "templates" / "values" / "defaults.yaml"
NOK8S_SCENARIO = REPO_ROOT / "config" / "scenarios" / "nok8s.yaml"


class _Logger:
    def log_info(self, *_: Any, **__: Any) -> None:
        pass

    log_warning = log_error = log_debug = log_info

    def line_break(self) -> None:
        pass


def _render(tmp_path: Path, cli_methods: str | None = None):
    return RenderPlans(
        template_dir=TEMPLATE_DIR,
        defaults_file=DEFAULTS_FILE,
        scenarios_file=NOK8S_SCENARIO,
        output_dir=tmp_path / "plan",
        logger=_Logger(),
        cli_methods=cli_methods,
    ).eval()


def _stack_dir(result) -> Path:
    paths = getattr(result, "rendered_paths", None) or []
    assert paths, "no rendered stacks produced"
    return Path(paths[0])


def test_nok8s_scenario_renders_templates_and_flags(tmp_path: Path) -> None:
    stack = _stack_dir(_render(tmp_path))

    # config.yaml: nok8s enabled, the other methods disabled.
    cfg = yaml.safe_load((stack / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["nok8s"]["enabled"] is True
    assert cfg["standalone"]["enabled"] is False
    assert cfg["modelservice"]["enabled"] is False
    assert cfg["kustomize"]["enabled"] is False
    assert cfg["fma"]["enabled"] is False

    # All four nok8s artifacts rendered with content.
    for prefix in (
        "31_nok8s-epp-config",
        "32_nok8s-epp-endpoints",
        "33_nok8s-envoy",
        "34_nok8s-containers",
    ):
        matches = list(stack.glob(f"{prefix}*"))
        assert matches, f"missing rendered file for {prefix}"
        assert matches[0].read_text(encoding="utf-8").strip(), f"{prefix} is empty"

    # endpoints file lists the single worker with the model label.
    endpoints = yaml.safe_load(
        next(stack.glob("32_nok8s-epp-endpoints*")).read_text(encoding="utf-8")
    )
    assert endpoints["endpoints"][0]["address"] == "127.0.0.1"
    assert endpoints["endpoints"][0]["port"] == "8000"
    assert endpoints["endpoints"][0]["labels"]["model"] == "Qwen/Qwen2.5-0.5B-Instruct"

    # container launch spec exposes the local endpoint.
    spec = yaml.safe_load(
        next(stack.glob("34_nok8s-containers*")).read_text(encoding="utf-8")
    )
    assert spec["endpoint"] == "http://localhost:8081"
    kinds = sorted(c["kind"] for c in spec["containers"])
    assert kinds == ["envoy", "epp", "vllm"]


def test_should_skip_selects_by_method() -> None:
    from llmdbenchmark.standup.steps.step_06_nok8s_deploy import NoK8sDeployStep
    from llmdbenchmark.run.steps.step_07_deploy_harness_local import (
        DeployHarnessLocalStep,
    )
    from llmdbenchmark.executor.context import ExecutionContext

    nok8s_ctx = ExecutionContext(
        plan_dir=Path("/tmp"), workspace=Path("/tmp"), deployed_methods=["nok8s"]
    )
    other_ctx = ExecutionContext(
        plan_dir=Path("/tmp"), workspace=Path("/tmp"), deployed_methods=["standalone"]
    )

    assert NoK8sDeployStep().should_skip(nok8s_ctx) is False
    assert NoK8sDeployStep().should_skip(other_ctx) is True

    assert DeployHarnessLocalStep().should_skip(other_ctx) is True
    assert DeployHarnessLocalStep().should_skip(nok8s_ctx) is False


class _FakeResult:
    def __init__(self, success: bool, stdout: str = "") -> None:
        self.success = success
        self.stdout = stdout
        self.stderr = ""
        self.exit_code = 0 if success else 1


class _FakeCmd:
    """Maps command substrings to success/failure for preflight testing."""

    def __init__(self, fail_substrings=(), stdout_for=None) -> None:
        self.fail_substrings = fail_substrings
        self.stdout_for = stdout_for or {}

    def execute(self, cmd, *_, **__):
        ok = not any(s in cmd for s in self.fail_substrings)
        out = ""
        for key, val in self.stdout_for.items():
            if key in cmd:
                out = val
        return _FakeResult(ok, out)


def _nok8s_ctx(tmp_path: Path, cmd):
    from llmdbenchmark.executor.context import ExecutionContext

    ctx = ExecutionContext(
        plan_dir=tmp_path,
        workspace=tmp_path,
        deployed_methods=["nok8s"],
        container_only=True,
        container_runtime="docker",
    )
    ctx.cmd = cmd
    ctx.logger = _Logger()
    return ctx


def test_nok8s_preflight_fatal_when_runtime_missing(tmp_path: Path) -> None:
    from llmdbenchmark.standup.steps.step_00_ensure_infra import EnsureInfraStep

    ctx = _nok8s_ctx(tmp_path, _FakeCmd(fail_substrings=("command -v docker",)))
    result = EnsureInfraStep().execute(ctx)
    assert result.success is False
    assert any("runtime" in e.lower() for e in result.errors)


def test_nok8s_preflight_passes_when_runtime_and_gpu_present(tmp_path: Path) -> None:
    from llmdbenchmark.standup.steps.step_00_ensure_infra import EnsureInfraStep

    # Everything succeeds; ss reports no listeners -> no busy ports.
    ctx = _nok8s_ctx(tmp_path, _FakeCmd(stdout_for={"ss -ltn": "State  Recv-Q\n"}))
    import os

    os.environ["HUGGING_FACE_HUB_TOKEN"] = "hf_test"
    try:
        result = EnsureInfraStep().execute(ctx)
    finally:
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    assert result.success is True


def test_resolve_deploy_method_forces_nok8s() -> None:
    """--methods nok8s wins and disables the other methods (mutual exclusion)."""
    rp = RenderPlans(
        template_dir=TEMPLATE_DIR,
        defaults_file=DEFAULTS_FILE,
        scenarios_file=NOK8S_SCENARIO,
        output_dir=Path("/tmp/unused-nok8s-plan"),
        logger=_Logger(),
        cli_methods="nok8s",
    )
    out = rp._resolve_deploy_method(
        {"standalone": {"enabled": True}, "modelservice": {"enabled": True}}
    )
    assert out["nok8s"]["enabled"] is True
    assert out["standalone"]["enabled"] is False
    assert out["modelservice"]["enabled"] is False
