"""Tests for auto-detected accelerator runtime profiles."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from llmdbenchmark.parser.cluster_resource_resolver import (
    ClusterResourceResolver,
    NodeResources,
)
from llmdbenchmark.parser.render_plans import RenderPlans
from llmdbenchmark.parser.version_resolver import VersionResolver


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = PROJECT_ROOT / "config" / "templates" / "jinja"
DEFAULTS = PROJECT_ROOT / "config" / "templates" / "values" / "defaults.yaml"
GUIDE = PROJECT_ROOT / "config" / "scenarios" / "guides" / "optimized-baseline.yaml"
XPU_SCENARIO = PROJECT_ROOT / "config" / "scenarios" / "examples" / "intel-xpu.yaml"
XPU_GUIDES = (
    "optimized-baseline.yaml",
    "pd-disaggregation.yaml",
    "precise-prefix-cache-routing.yaml",
)


def _render(
    tmp_path: Path,
    profile: str,
    resource: str | None,
    guide: Path = GUIDE,
    setup_overrides: dict | None = None,
) -> tuple[object, dict]:
    logger = MagicMock()
    accelerator_override = {"profile": profile}
    if resource is not None:
        accelerator_override["resource"] = resource
    overrides = {"accelerator": accelerator_override}
    if setup_overrides:
        overrides.update(setup_overrides)
    renderer = RenderPlans(
        template_dir=TEMPLATES,
        defaults_file=DEFAULTS,
        scenarios_file=guide,
        output_dir=tmp_path,
        logger=logger,
        setup_overrides=overrides,
        version_resolver=VersionResolver(logger=logger, dry_run=True),
        cluster_resource_resolver=ClusterResourceResolver(logger=logger, dry_run=True),
    )
    result = renderer.eval()
    assert len(result.rendered_paths) == 1
    config_path = result.rendered_paths[0] / "config.yaml"
    assert config_path.exists()
    return result, yaml.safe_load(config_path.read_text())


def test_intel_xe_resource_resolves_profile_and_type():
    resolver = ClusterResourceResolver(logger=MagicMock(), dry_run=False)
    resolver._node_resources = NodeResources(accelerator_resources=["gpu.intel.com/xe"])
    values = {"accelerator": {"resource": "auto", "profile": "auto"}}
    unresolved: list[str] = []

    resolver._resolve_accelerator_resource(values, unresolved)
    resolver._resolve_accelerator_profile(values, unresolved)

    assert unresolved == []
    assert values["accelerator"] == {
        "resource": "gpu.intel.com/xe",
        "profile": "intel-xe",
        "type": "intel-xe",
    }


def test_intel_i915_and_xe_aliases_prefer_xe():
    resolver = ClusterResourceResolver(logger=MagicMock(), dry_run=False)
    resolver._node_resources = NodeResources(
        accelerator_resources=["gpu.intel.com/i915", "gpu.intel.com/xe"]
    )
    values = {"accelerator": {"resource": "auto", "profile": "auto"}}
    unresolved: list[str] = []

    resolver._resolve_accelerator_resource(values, unresolved)
    resolver._resolve_accelerator_profile(values, unresolved)

    assert unresolved == []
    assert values["accelerator"]["resource"] == "gpu.intel.com/xe"
    assert values["accelerator"]["profile"] == "intel-xe"


def test_intel_i915_uses_the_shared_xpu_profile(tmp_path):
    result, merged = _render(tmp_path, "intel-i915", "gpu.intel.com/i915")

    assert not result.has_errors
    assert merged["accelerator"]["type"] == "intel-i915"
    assert merged["accelerator"]["resource"] == "gpu.intel.com/i915"
    assert "llm-d-xpu" in merged["images"]["vllm"]["repository"]
    # The profile contributes machine values -- image, PVC size, resource
    # sizing -- and not a model: the guide's command names Qwen/Qwen3-32B, and
    # the model the engine is told to serve is the model the plan is built for.
    # Hardware that cannot hold it gets its own scenario file with its own
    # command (config/scenarios/examples/intel-xpu.yaml).
    assert merged["model"]["name"] == "Qwen/Qwen3-32B"
    assert merged["storage"]["modelPvc"]["size"] == "50Gi"


def test_explicit_profile_resolves_resource_without_cluster():
    resolver = ClusterResourceResolver(logger=MagicMock(), dry_run=False)

    resolved = resolver.resolve_all(
        {"accelerator": {"profile": "intel-xe", "resource": "auto"}}
    )

    assert resolved["accelerator"]["resource"] == "gpu.intel.com/xe"
    assert resolved["accelerator"]["type"] == "intel-xe"
    assert resolver._connected is False


def test_explicit_unified_xpu_profile_resolves_dra_driver_without_cluster():
    resolver = ClusterResourceResolver(logger=MagicMock(), dry_run=False)

    resolved = resolver.resolve_all(
        {"accelerator": {"profile": "intel-xpu", "resource": "auto"}}
    )

    assert "resource" not in resolved["accelerator"]
    assert resolved["accelerator"]["draDriver"] == "gpu.intel.com"
    assert resolved["accelerator"]["type"] == "intel-xpu"
    assert resolver._connected is False


def test_unified_xpu_dra_profile_uses_shared_xpu_overlay(tmp_path):
    result, merged = _render(tmp_path, "intel-xpu", None)

    assert not result.has_errors
    assert merged["accelerator"]["type"] == "intel-xpu"
    assert "resource" not in merged["accelerator"]
    assert merged["accelerator"]["draDriver"] == "gpu.intel.com"
    assert "llm-d-xpu" in merged["images"]["vllm"]["repository"]
    assert merged["model"]["name"] == "Qwen/Qwen3-32B"
    assert merged["dra"]["enabled"] is True
    assert merged["dra"]["claimTemplates"] == {"intel-xpu": {"class": "gpu.intel.com"}}

    plan_dir = result.rendered_paths[0]
    ms_values = yaml.safe_load((plan_dir / "13_ms-values.yaml").read_text())
    assert ms_values["accelerator"]["dra"] is True
    assert ms_values["accelerator"]["resourceClaimTemplates"] == {
        "intel-xpu": {"class": "gpu.intel.com"}
    }

    for rendered in ("13_ms-values.yaml", "14_standalone-deployment_yaml.yaml"):
        # A DRA driver is not an extended resource, so it must never appear
        # as a container resource key.
        assert 'gpu.intel.com: "' not in (plan_dir / rendered).read_text()


def test_cluster_connection_uses_cli_kubeconfig(monkeypatch):
    observed: dict[str, str | None] = {}
    api_client = object()

    def fake_kube_connect(kubeconfig=None, **_kwargs):
        observed["kubeconfig"] = kubeconfig
        return api_client

    monkeypatch.setattr(
        "llmdbenchmark.utilities.cluster.kube_connect", fake_kube_connect
    )
    monkeypatch.setattr("llmdbenchmark.utilities.cluster._KUBE_AVAILABLE", True)

    resolver = ClusterResourceResolver(
        logger=MagicMock(), kubeconfig="/tmp/test-kubeconfig"
    )

    assert resolver._connect(["accelerator.resource"])
    assert resolver._api_client is api_client
    assert observed["kubeconfig"] == "/tmp/test-kubeconfig"


def test_multiple_accelerators_require_explicit_selection():
    resolver = ClusterResourceResolver(logger=MagicMock(), dry_run=False)
    resolver._node_resources = NodeResources(
        accelerator_resources=["gpu.intel.com/xe", "nvidia.com/gpu"]
    )
    values = {"accelerator": {"resource": "auto", "profile": "auto"}}

    try:
        resolver._resolve_accelerator_resource(values, [])
    except RuntimeError as exc:
        assert "Multiple accelerator resources" in str(exc)
    else:
        raise AssertionError("ambiguous accelerator selection must fail")


def test_same_guide_uses_intel_runtime_profile(tmp_path):
    result, merged = _render(tmp_path, "intel-xe", "gpu.intel.com/xe")

    assert not result.has_errors
    assert merged["accelerator"]["type"] == "intel-xe"
    assert "llm-d-xpu" in merged["images"]["vllm"]["repository"]
    # The profile no longer names a model: it contributes what the machine
    # decides (image, resource and storage sizing) and the guide command names
    # what is served.
    assert merged["model"]["name"] == "Qwen/Qwen3-32B"
    # The profile's `blockSize: 16` is a fallback, not an override: it is the
    # page size an XPU command leaves unsaid, and this guide's command says
    # `--block-size 64`. The command reaches the engine either way, so the plan
    # follows it -- a profile cannot move a number it cannot edit.
    assert merged["model"]["blockSize"] == 64
    assert merged["decode"]["resources"]["limits"] == {
        "memory": "24Gi",
        "cpu": "8",
    }
    assert merged["decode"]["resources"]["requests"] == {
        "memory": "12Gi",
        "cpu": "4",
    }
    assert merged["decode"]["parallelism"]["tensor"] == 1
    assert merged["storage"]["modelPvc"]["size"] == "50Gi"
    assert merged["storage"]["modelPvc"]["accessModes"] == ["ReadWriteOnce"]
    assert merged["storage"]["workloadPvc"]["accessModes"] == ["ReadWriteOnce"]
    # Keep the constrained-hardware request at 8Gi, while allowing enough
    # limit headroom for CI's 64Gi harness request override.
    assert merged["harness"]["resources"] == {
        "cpu": 2,
        "memory": "8Gi",
        "memoryLimit": "64Gi",
    }
    # An accelerator profile contributes values, never command text. It cannot
    # reach into the guide's command, so the guide renders the command it spells
    # out -- flags and all -- on every backend. Hardware that needs a different
    # command is its own scenario file (see test_xpu_scenario_* below).
    for fragment_key in (
        "runtimePreamble",
        "dtypeArgs",
        "executionArgs",
        "memoryUtilizationArgs",
        "blockSizeArgs",
        "kvBufferDeviceJson",
    ):
        assert fragment_key not in merged["accelerator"]
    assert "${accelerator." not in merged["decode"]["engine"]["command"]

    modelservice_values = (result.rendered_paths[0] / "13_ms-values.yaml").read_text()
    assert "gpu.intel.com/xe" in modelservice_values
    assert "ghcr.io/llm-d/llm-d-xpu" in modelservice_values
    assert "supplementalGroups:" not in modelservice_values
    assert "--dtype bfloat16" in modelservice_values


def test_xpu_profile_keeps_precise_router_compact_and_token_optional(tmp_path):
    guide = (
        PROJECT_ROOT
        / "config"
        / "scenarios"
        / "guides"
        / "precise-prefix-cache-routing.yaml"
    )
    result, merged = _render(tmp_path, "intel-xe", "gpu.intel.com/xe", guide)

    assert not result.has_errors
    # assert merged["router"]["epp"]["env"] == []
    assert merged["router"]["epp"]["resources"]["requests"]["cpu"] == "1"
    assert merged["router"]["proxy"]["resources"]["requests"]["cpu"] == "1"
    assert "$(POD_IP):$(ENGINE_PORT)" in merged["decode"]["engine"]["command"]
    assert "POD_PORT" not in merged["decode"]["engine"]["command"]
    # The engine's page size and the EPP token processor's must agree. One plan
    # fact feeds both, and it is read off the command -- this guide's command
    # states `--block-size 64`, so both get 64 even though the XPU profile
    # states 16: a number the engine was never given cannot be the one the
    # router hashes on.
    assert merged["model"]["blockSize"] == 64
    plugins = merged["router"]["epp"]["pluginsCustomConfig"][
        "precise-prefix-cache-routing-plugins.yaml"
    ]
    assert "blockSizeTokens: 64" in plugins


def test_standalone_without_accelerator_labels_uses_resource_scheduling(tmp_path):
    result, _ = _render(
        tmp_path,
        "intel-xe",
        "gpu.intel.com/xe",
        PROJECT_ROOT / "config" / "scenarios" / "examples" / "gpu.yaml",
        setup_overrides={
            "modelservice": {"enabled": False},
            "standalone": {
                "enabled": True,
                "acceleratorType": {"labelKey": "", "labelValue": ""},
            },
        },
    )

    assert not result.has_errors
    deployment_path = result.rendered_paths[0] / "14_standalone-deployment_yaml.yaml"
    deployment_text = deployment_path.read_text()
    deployment = yaml.safe_load(deployment_text)
    pod_spec = deployment["spec"]["template"]["spec"]
    assert "affinity" not in pod_spec
    assert "gpu.intel.com/xe" in deployment_text
    assert "nvidia.com/gpu:None:None" not in deployment_text


@pytest.mark.parametrize("guide_name", XPU_GUIDES)
def test_all_supported_guides_render_from_their_canonical_file(tmp_path, guide_name):
    guide = PROJECT_ROOT / "config" / "scenarios" / "guides" / guide_name
    result, merged = _render(
        tmp_path / guide.stem,
        "intel-xe",
        "gpu.intel.com/xe",
        guide,
    )

    assert not result.has_errors
    assert merged["accelerator"]["profile"] == "intel-xe"
    assert "llm-d-xpu" in merged["images"]["vllm"]["repository"]
    # Each guide serves the model its own command names. These three name
    # Qwen/Qwen3-32B, which a single B60 does not hold -- rendering on the XPU
    # profile is what this test covers, not running there. The supported XPU
    # path is config/scenarios/examples/intel-xpu.yaml.
    assert merged["model"]["name"] == "Qwen/Qwen3-32B"
    # Each guide's own `--block-size`, not the profile's fallback -- see
    # test_same_guide_uses_intel_runtime_profile.
    assert merged["model"]["blockSize"] == (
        128 if guide_name == "pd-disaggregation.yaml" else 64
    )


def test_same_guide_keeps_nvidia_configuration(tmp_path):
    result, merged = _render(tmp_path, "nvidia", "nvidia.com/gpu")

    assert not result.has_errors
    assert merged["accelerator"]["type"] == "nvidia"
    assert "llm-d-xpu" not in merged["images"]["vllm"]["repository"]
    assert merged["model"]["name"] == "Qwen/Qwen3-32B"
    # Finding the injected CUDA driver is the one runtime fact no file can state
    # at render time. It is emitted into /shared-config/llmdbench_env.sh by the
    # preprocess step and sourced in the same shell, so it never appears in --
    # and never has to be interpolated into -- the command itself.
    assert "libcuda.so.1" not in merged["decode"]["engine"]["command"]
    assert "--dtype bfloat16" in merged["decode"]["engine"]["command"]
    assert "--block-size 64" in merged["decode"]["engine"]["command"]
    assert "--gpu-memory-utilization 0.95" in merged["decode"]["engine"]["command"]
    assert "--enforce-eager" not in merged["decode"]["engine"]["command"]
    # Nothing llm-d-benchmark owns is left in the command it hands the engine.
    assert "VLLM_" not in merged["decode"]["engine"]["command"]


# ---------------------------------------------------------------------------
# The XPU scenario file
#
# An accelerator profile supplies values (image, a fitting model, resource and
# storage sizing). It supplies no command text, so hardware that needs a
# materially different command states that command in its own scenario file --
# the way examples/spyre.yaml and examples/cpu.yaml always have. These tests are
# that file's contract.
# ---------------------------------------------------------------------------


def _render_scenario(tmp_path: Path, scenario: Path) -> tuple[object, Path, dict]:
    """Render a scenario with no overrides at all: it must stand on its own."""
    logger = MagicMock()
    renderer = RenderPlans(
        template_dir=TEMPLATES,
        defaults_file=DEFAULTS,
        scenarios_file=scenario,
        output_dir=tmp_path,
        logger=logger,
        version_resolver=VersionResolver(logger=logger, dry_run=True),
        cluster_resource_resolver=ClusterResourceResolver(logger=logger, dry_run=True),
    )
    result = renderer.eval()
    assert not result.has_errors
    assert len(result.rendered_paths) == 1
    plan_dir = result.rendered_paths[0]
    return result, plan_dir, yaml.safe_load((plan_dir / "config.yaml").read_text())


def test_xpu_scenario_states_its_engine_flags_literally(tmp_path):
    """The four XPU differences are readable in the file, not interpolated."""
    _, _, merged = _render_scenario(tmp_path, XPU_SCENARIO)

    command = merged["decode"]["engine"]["command"]
    assert "--enforce-eager" in command
    assert "--disable-sliding-window" in command
    assert "--gpu-memory-utilization 0.35" in command
    # The XPU FMHA kernel picks its own page size; passing the canonical CUDA
    # value fails on first inference.
    assert "--block-size" not in command
    # Nothing llm-d-benchmark owns is left in what the engine is handed.
    assert "${" not in command
    assert "VLLM_" not in command


def test_xpu_scenario_takes_values_from_the_shared_profile(tmp_path):
    """It states no image, no sizing and no storage: `intel-xe` does.

    The model is the other way round. A profile no longer names one, so the
    scenario's own command is where Qwen/Qwen3-0.6B comes from, and the only
    model key the file writes down is the Kubernetes name prefix.
    """
    _, plan_dir, merged = _render_scenario(tmp_path, XPU_SCENARIO)

    assert merged["accelerator"]["type"] == "intel-xe"
    assert merged["accelerator"]["resource"] == "gpu.intel.com/xe"
    assert "llm-d-xpu" in merged["images"]["vllm"]["repository"]
    assert merged["model"]["name"] == "Qwen/Qwen3-0.6B"
    assert merged["model"]["shortName"] == "qwen-qwen3-06b"
    assert merged["model"]["huggingfaceId"] == "Qwen/Qwen3-0.6B"
    assert merged["model"]["path"] == "models/Qwen/Qwen3-0.6B"
    assert merged["decode"]["resources"]["limits"] == {"memory": "24Gi", "cpu": "8"}
    assert merged["storage"]["modelPvc"]["size"] == "50Gi"
    assert merged["harness"]["resources"]["memory"] == "8Gi"

    modelservice_values = (plan_dir / "13_ms-values.yaml").read_text()
    assert "gpu.intel.com/xe" in modelservice_values
    assert "ghcr.io/llm-d/llm-d-xpu" in modelservice_values


def test_xpu_scenario_capacity_tracks_the_command(tmp_path):
    """The planner's KV-cache arithmetic uses the numbers the engine received.

    The profile deliberately states neither ``maxModelLen`` nor
    ``gpuMemoryUtilization``: restating a number the command already carries is
    how the planner ends up reasoning about a pod that was never launched.
    ``blockSize`` is the one number the profile does state, because an XPU
    command carries no page size at all -- the FMHA kernel fixes it at 16 -- and
    the prefix-cache router has to agree with the engine about it. It is a
    fallback for that silence: a command that does state a page size is the one
    the engine gets, so the plan follows the command instead.
    """
    _, _, merged = _render_scenario(tmp_path, XPU_SCENARIO)

    assert merged["model"]["gpuMemoryUtilization"] == 0.35
    assert merged["model"]["maxModelLen"] == 16000
    assert merged["model"]["blockSize"] == 16
    # The pair above is all that is read: `--max-num-seqs` is the engine's
    # business and has no key in the plan.
    assert set(merged["model"]) & {"maxNumSeq", "maxNumBatchedTokens"} == set()


def test_xpu_scenario_engine_command_never_mentions_the_env_file(tmp_path):
    """`/shared-config/llmdbench_env.sh` and the CUDA preamble are preprocess
    concerns; the command line is the engine's own."""
    _, plan_dir, merged = _render_scenario(tmp_path, XPU_SCENARIO)

    command = merged["decode"]["engine"]["command"]
    assert "llmdbench_env.sh" not in command
    assert "libcuda.so.1" not in command
    assert "LD_LIBRARY_PATH" not in command
    # It still runs, ahead of the command, in the same shell.
    launch = yaml.safe_load((plan_dir / "13_ms-values.yaml").read_text())["decode"][
        "containers"
    ][0]["args"][-1]
    assert "set_llmdbench_environment.py" in launch
    assert launch.index("set_llmdbench_environment.py") < launch.index("vllm serve")
