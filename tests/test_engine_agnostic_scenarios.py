"""Tests that a non-vLLM engine needs a different command and nothing else.

Supporting SGLang or TensorRT-LLM takes no new scenario keys: the user pastes
the launch line they already run, and the Kubernetes facts that hang off the
process -- image, container port, probes, Service wiring, the capacity check's
two inputs -- follow from it. What the command cannot answer is stated in
Kubernetes' own vocabulary, not in the engine's: how many devices a pod holds
is granted by the kubelet before the process exists.

``config/scenarios/examples/engines.yaml`` is that claim written down: one
stack, one active vLLM command, and the SGLang and TensorRT-LLM commands beside
it as ``# @engine``-tagged alternatives. These tests apply each tag -- the same
switch ``--engine <name>`` performs -- render the result, and assert both halves:
what the command supplies, and what deliberately does not come from it. If a
future change makes an engine need its own configuration key again, one of these
fails.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from llmdbenchmark.engine import switch_scenario_file
from llmdbenchmark.parser.cluster_resource_resolver import ClusterResourceResolver
from llmdbenchmark.parser.render_plans import RenderPlans
from llmdbenchmark.parser.version_resolver import VersionResolver


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = PROJECT_ROOT / "config" / "templates" / "jinja"
DEFAULTS = PROJECT_ROOT / "config" / "templates" / "values" / "defaults.yaml"
EXAMPLES = PROJECT_ROOT / "config" / "scenarios" / "examples"
ENGINES = EXAMPLES / "engines.yaml"


def _scenario(tmp_path: Path, engine: str) -> Path:
    """``examples/engines.yaml`` switched to ``engine``.

    The same edit ``llmdbenchmark --engine <engine>`` makes: uncomment that
    engine's tagged groups and drop the definitions they replace. Written to a
    directory of its own so it cannot be mistaken for rendered output.
    """
    out = tmp_path / "scenario"
    switched = switch_scenario_file(ENGINES, engine, out)
    assert switched is not None, f"engines.yaml already launches {engine}"
    return switched


def _render(tmp_path: Path, scenario: Path):
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
    merged = yaml.safe_load((plan_dir / "config.yaml").read_text())
    return result, plan_dir, merged


def _resolve(tmp_path: Path, engine: str) -> Path:
    """The scenario as ``--engine <engine>`` would render it, vLLM included.

    vLLM is the active command, so it is the file itself -- which is what makes
    the shared assertions below cover all three engines, not just the two that
    arrive by a switch.
    """
    return ENGINES if engine == "vllm" else _scenario(tmp_path, engine)


def _warnings(result) -> list[str]:
    out: list[str] = []
    for stack in result.stacks.values():
        for attr in ("validation_warnings", "render_errors", "missing_fields"):
            out.extend(getattr(stack, attr) or [])
    return out


# ---------------------------------------------------------------------------
# SGLang
# ---------------------------------------------------------------------------


def test_sglang_example_renders_clean(tmp_path):
    result, _, _ = _render(tmp_path, _scenario(tmp_path, "sglang"))

    # A command in another engine's spelling must not trip the engine-mismatch
    # or unknown-launcher advisories.
    assert _warnings(result) == []


def test_sglang_image_follows_the_command(tmp_path):
    """The scenario names no image: the launcher identifies SGLang, and the
    engine selects ``images.sglang``."""
    _, _, merged = _render(tmp_path, _scenario(tmp_path, "sglang"))

    repository = merged["decode"]["engine"]["image"]["repository"]
    assert "sglang" in repository
    assert "vllm" not in repository


def test_sglang_capacity_is_read_in_sglangs_own_spelling(tmp_path):
    """Two numbers are read back out of the command, in SGLang's spelling.

    ``--context-length`` and ``--mem-fraction-static`` are what vLLM calls
    ``--max-model-len`` and ``--gpu-memory-utilization``, and the pre-deploy
    capacity check sizes KV cache against them, so they are read. So is
    ``--page-size``, which is what vLLM calls ``--block-size``: a prefix-cache
    router has to hash on the engine's own page boundaries. Nothing else is --
    the batch widths belong to the engine alone, and the scenario's ``model:``
    block restates none of it."""
    _, _, merged = _render(tmp_path, _scenario(tmp_path, "sglang"))

    assert merged["model"]["maxModelLen"] == 32768
    assert merged["model"]["gpuMemoryUtilization"] == 0.9
    assert merged["model"]["blockSize"] == 64
    # --max-running-requests and --max-prefill-tokens have no key to land in.
    assert set(merged["model"]) & {"maxNumSeq", "maxNumBatchedTokens"} == set()


def test_sglang_port_comes_from_the_command(tmp_path):
    """``--port`` is spelled identically by every supported engine, and the
    Service has to pick an endpoint before the process exists -- so this one
    flag is read, at no per-engine cost."""
    _, _, merged = _render(tmp_path, _scenario(tmp_path, "sglang"))

    # --port 8200: the routing sidecar owns the Service port and forwards here.
    assert merged["decode"]["engine"]["port"] == 8200
    assert merged["engine"]["servicePort"] == 8000


def test_sglang_device_count_is_a_kubernetes_fact(tmp_path):
    """``--tp-size`` is not read: the kubelet grants devices before the engine
    exists, so the count is stated in Kubernetes' vocabulary. This role states
    nothing, and gets the chart's single-pod width."""
    _, plan_dir, merged = _render(tmp_path, _scenario(tmp_path, "sglang"))

    assert "acceleratorCount" not in merged["decode"]["engine"]
    limits = yaml.safe_load((plan_dir / "13_ms-values.yaml").read_text())["decode"][
        "containers"
    ][0]["resources"]["limits"]
    assert limits[merged["accelerator"]["resource"]] == "1"


def test_sglang_command_reaches_the_container_verbatim(tmp_path):
    _, plan_dir, merged = _render(tmp_path, _scenario(tmp_path, "sglang"))

    container = yaml.safe_load((plan_dir / "13_ms-values.yaml").read_text())["decode"][
        "containers"
    ][0]
    launch = container["args"][-1]

    assert container["name"] == "modelserver"
    assert container["command"] == ["/bin/bash", "-c"]
    # Every flag the user wrote, in the spelling they wrote it.
    for flag in (
        "python3 -m sglang.launch_server",
        "--model-path",
        "--tp-size 1",
        "--context-length 32768",
        "--page-size 64",
        "--mem-fraction-static 0.9",
        "--max-running-requests 256",
        "--max-prefill-tokens 8192",
        "--enable-metrics",
    ):
        assert flag in launch, flag
    # Nothing llm-d-benchmark owns leaks into the engine's command line.
    assert "--max-model-len" not in launch
    assert "VLLM_" not in launch


# ---------------------------------------------------------------------------
# TensorRT-LLM
# ---------------------------------------------------------------------------


def test_trtllm_example_renders_clean(tmp_path):
    result, _, _ = _render(tmp_path, _scenario(tmp_path, "trtllm"))

    assert _warnings(result) == []


def test_trtllm_image_follows_the_command(tmp_path):
    _, _, merged = _render(tmp_path, _scenario(tmp_path, "trtllm"))

    repository = merged["decode"]["engine"]["image"]["repository"]
    assert "tensorrt" in repository.lower()


def test_trtllm_capacity_is_read_from_underscore_flags(tmp_path):
    """TRT-LLM spells its flags with underscores -- ``--max_seq_len`` and
    ``--free_gpu_memory_fraction`` -- and the capacity pair is read in that
    spelling. ``blockSize`` has no CLI flag at all (``tokens_per_block`` lives
    in the ``--extra_llm_api_options`` YAML), so the scenario states it and the
    command writes it into that file from the same plan value."""
    _, plan_dir, merged = _render(tmp_path, _scenario(tmp_path, "trtllm"))

    assert merged["model"]["maxModelLen"] == 32768
    assert merged["model"]["gpuMemoryUtilization"] == 0.9
    assert set(merged["model"]) & {"maxNumSeq", "maxNumBatchedTokens"} == set()

    block_size = merged["model"]["blockSize"]
    assert block_size == 32
    launch = yaml.safe_load((plan_dir / "13_ms-values.yaml").read_text())["decode"][
        "containers"
    ][0]["args"][-1]
    assert f"tokens_per_block: {block_size}" in launch


def test_trtllm_subcommand_spelling_still_yields_the_model(tmp_path):
    """The guide spells it ``trtllm-serve serve <model>``; the subcommand must
    not be read as the model reference, or ``--models`` on the CLI breaks."""
    result, _, _ = _render(tmp_path, _scenario(tmp_path, "trtllm"))

    assert not any("does not reference the model" in w for w in _warnings(result))


def test_trtllm_port_comes_from_the_command(tmp_path):
    _, _, merged = _render(tmp_path, _scenario(tmp_path, "trtllm"))

    assert merged["decode"]["engine"]["port"] == 8200
    assert merged["engine"]["servicePort"] == 8000


def test_trtllm_runtime_env_reaches_the_container(tmp_path):
    """TRT-LLM's wheel needs its bundled TensorRT libraries on the loader path,
    which the llm-d guide sets the same way -- via the pod env, not a flag."""
    _, plan_dir, _ = _render(tmp_path, _scenario(tmp_path, "trtllm"))

    container = yaml.safe_load((plan_dir / "13_ms-values.yaml").read_text())["decode"][
        "containers"
    ][0]
    env = {e["name"]: e.get("value") for e in container.get("env", [])}

    assert "/usr/local/tensorrt/lib" in (env.get("LD_LIBRARY_PATH") or "")


# ---------------------------------------------------------------------------
# what the two scenarios have in common
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["vllm", "sglang", "trtllm"])
def test_no_engine_specific_configuration_keys(tmp_path, name):
    """The only engine-specific text in a scenario is the command itself.

    No ``vllm:``/``sglang:``/``trtllm:`` block, no per-engine flag keys."""
    text = _resolve(tmp_path, name).read_text()
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    scenario = yaml.safe_load(body)["scenario"][0]

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in ("vllm", "sglang", "trtllm", "vllmCommon"), (
                    f"{name}: engine-specific block at {path}.{key}"
                )
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(scenario)


@pytest.mark.parametrize("name", ["vllm", "sglang", "trtllm"])
def test_model_id_is_read_off_the_command(tmp_path, name):
    """The command names the model the way a user would on a node -- a plain
    Hugging Face id, written once -- and every model fact outside the engine is
    read back off it rather than typed a second time.

    The scenario's own ``model:`` block carries neither the id nor the path: the
    name, the hub id and the PVC subdirectory all come from that one literal, so
    they cannot disagree with what the engine is actually serving.
    """
    _, _, merged = _render(tmp_path, _resolve(tmp_path, name))

    command = merged["decode"]["engine"]["command"]
    assert "Qwen/Qwen3-0.6B" in command
    assert merged["model"]["name"] == "Qwen/Qwen3-0.6B"
    assert merged["model"]["huggingfaceId"] == "Qwen/Qwen3-0.6B"
    assert merged["model"]["path"] == "models/Qwen/Qwen3-0.6B"
    # Nothing rewrote the command to get there.
    assert "--served-model-name" not in command
