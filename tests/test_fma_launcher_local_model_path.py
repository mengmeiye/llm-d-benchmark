"""What an FMA arm states, and what it inherits (#1822).

Fast Model Actuation is the one path with no ``vllm serve`` line to write down:
the launcher process *is* the engine entrypoint -- it is what owns
``--enable-sleep-mode`` -- so the CRD takes a flag string rather than a command.
``fma.launcher.options`` is that string, this path's counterpart of a role's
``engine.command``, and like a command it is passed through verbatim. Nothing
assembles it flag by flag, so nothing here has to be taught a new vLLM flag.

Two things still have to hold, and they are what these tests pin:

* #1822 -- an FMA arm can load the weights the download job already staged on
  the PVC instead of re-resolving the repo ID through the HF hub cache (a
  second, network-dependent copy that can re-download at weight load). That is
  written as ``--model ${fma.modelMountPath}/${model.path}``, and because a
  local path would mangle the derived Helm release name, the advertised ID is
  pinned back with ``--served-model-name ${model.name}``.
* Parity -- the numbers an FMA arm must share with the arm it is compared
  against go in as ``${model.*}`` references, so the two cannot desync.

Both are ordinary ``${...}`` references, resolved by the same substitution every
other scenario value goes through, which is why neither needs a key of its own.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import yaml

from llmdbenchmark.parser.cluster_resource_resolver import ClusterResourceResolver
from llmdbenchmark.parser.render_plans import RenderPlans
from llmdbenchmark.parser.version_resolver import VersionResolver


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = PROJECT_ROOT / "config" / "templates" / "jinja"
DEFAULTS = PROJECT_ROOT / "config" / "templates" / "values" / "defaults.yaml"
SCENARIO = PROJECT_ROOT / "config" / "scenarios" / "examples" / "fma.yaml"


def _render(tmp_path: Path, overrides: dict) -> tuple[dict, str]:
    """Render the FMA example and return ``(merged config, rendered options)``."""
    logger = MagicMock()
    renderer = RenderPlans(
        template_dir=TEMPLATES,
        defaults_file=DEFAULTS,
        scenarios_file=SCENARIO,
        output_dir=tmp_path,
        logger=logger,
        setup_overrides=overrides,
        version_resolver=VersionResolver(logger=logger, dry_run=True),
        cluster_resource_resolver=ClusterResourceResolver(logger=logger, dry_run=True),
    )
    result = renderer.eval()
    assert not result.has_errors
    plan_dir = result.rendered_paths[0]
    merged = yaml.safe_load((plan_dir / "config.yaml").read_text())
    docs = [
        doc
        for doc in yaml.safe_load_all((plan_dir / "24_fma-deployment.yaml").read_text())
        if doc
    ]
    isc = next(doc for doc in docs if doc.get("kind") == "InferenceServerConfig")
    return merged, isc["spec"]["modelServerConfig"]["options"]


def _options_override(options: str, **model: object) -> dict:
    override: dict = {"fma": {"launcher": {"options": options}}}
    if model:
        override["model"] = model
    return override


class TestStagedLocalWeights:
    """#1822, written the way a user writes it: in the flag string."""

    def test_staged_path_and_pinned_id_render_verbatim(self, tmp_path):
        merged, options = _render(
            tmp_path,
            _options_override(
                "--model ${fma.modelMountPath}/${model.path} "
                "--served-model-name ${model.name} --enable-sleep-mode"
            ),
        )

        mount = merged["fma"]["modelMountPath"]
        assert options == (
            f"--model {mount}/{merged['model']['path']} "
            f"--served-model-name {merged['model']['name']} --enable-sleep-mode"
        )

    def test_the_advertised_id_stays_the_repo_id(self, tmp_path):
        """``model.name`` feeds the derived Helm release name, which a local
        path mangles into an invalid (leading-dash) name. Naming the path in
        ``--model`` must not touch it."""
        merged, options = _render(
            tmp_path,
            _options_override(
                "--model ${fma.modelMountPath}/${model.path} "
                "--served-model-name ${model.name}"
            ),
        )

        assert merged["model"]["name"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert f"--served-model-name {merged['model']['name']}" in options


class TestCapacityParity:
    def test_capacity_references_resolve_to_the_plans_numbers(self, tmp_path):
        """One number, one place: whatever the plan holds is what both arms get."""
        merged, options = _render(
            tmp_path,
            _options_override(
                "--model ${model.name} --enable-sleep-mode "
                "--max-model-len ${model.maxModelLen} "
                "--block-size ${model.blockSize} "
                "--gpu-memory-utilization ${model.gpuMemoryUtilization}",
                maxModelLen=16384,
                blockSize=64,
                gpuMemoryUtilization=0.9,
            ),
        )

        assert merged["model"]["maxModelLen"] == 16384
        assert options == (
            "--model meta-llama/Llama-3.1-8B-Instruct --enable-sleep-mode "
            "--max-model-len 16384 --block-size 64 --gpu-memory-utilization 0.9"
        )


class TestNothingIsAssembled:
    def test_no_flag_is_added_on_the_users_behalf(self, tmp_path):
        """Not the model, not prefix caching, not a log-level -- nothing.

        An FMA arm that wants a flag writes it, exactly as every other path
        writes flags into ``engine.command``.
        """
        _, options = _render(tmp_path, _options_override("--enable-sleep-mode"))

        assert options == "--enable-sleep-mode"
