"""How a scenario says where the weights come from, and what that renders.

One key decides it -- ``modelservice.uriProtocol`` -- and the engine command is
the same line under every value of it: the plain Hugging Face id, exactly as it
would be typed on a node. That is the point of the default, ``pvc+hf``: the PVC
holds a Hugging Face *hub cache*, the pod's ``HF_HUB_CACHE`` points into it, and
a bare id resolves against staged weights instead of reaching for the Hub. The
command does not have to name a path, so it survives a protocol switch.

These tests pin the three protocols end to end -- the uri the chart is handed,
the environment the pods get, and the directory the download job writes -- plus
the one shape ``pvc+hf`` cannot serve.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from llmdbenchmark.parser.cluster_resource_resolver import ClusterResourceResolver
from llmdbenchmark.parser.render_plans import RenderPlans
from llmdbenchmark.parser.version_resolver import VersionResolver
from llmdbenchmark.standup.steps.step_04_model_namespace import (
    ModelNamespaceStep,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = PROJECT_ROOT / "config" / "templates" / "jinja"
DEFAULTS = PROJECT_ROOT / "config" / "templates" / "values" / "defaults.yaml"
SCENARIO = PROJECT_ROOT / "config" / "scenarios" / "examples" / "gpu.yaml"


def _render(
    tmp_path: Path,
    overrides: dict | None = None,
    scenario: Path = SCENARIO,
    expect_errors: bool = False,
):
    logger = MagicMock()
    renderer = RenderPlans(
        template_dir=TEMPLATES,
        defaults_file=DEFAULTS,
        scenarios_file=scenario,
        output_dir=tmp_path,
        logger=logger,
        setup_overrides=overrides or {},
        version_resolver=VersionResolver(logger=logger, dry_run=True),
        cluster_resource_resolver=ClusterResourceResolver(logger=logger, dry_run=True),
    )
    result = renderer.eval()
    if expect_errors:
        # A stack that fails validation renders nothing, so there is no plan
        # directory to read back -- only the errors on the stack.
        return result, None, None
    assert len(result.rendered_paths) == 1
    plan_dir = result.rendered_paths[0]
    merged = yaml.safe_load((plan_dir / "config.yaml").read_text())
    return result, plan_dir, merged


def _chart_hub_cache_subdir(artifacts: dict) -> str:
    """The cache subdirectory the modelservice chart derives from our uri.

    The chart's ``hfEnv`` helper splits the uri's path on ``/``, drops the claim
    name at the front and the ``<org>/<model>`` id at the back, and joins what is
    left onto its own mount path. Reimplemented here so the uri we hand the chart
    is checked against the chart's own arithmetic rather than against our copy of
    the answer.
    """
    claimpath = artifacts["uri"].split("://", 1)[1].split("/")
    return "/".join(claimpath[1:-2])


def _ms_values(plan_dir: Path) -> dict:
    return yaml.safe_load((plan_dir / "13_ms-values.yaml").read_text())


def _env(container: dict) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in container.get("env") or []:
        out[item["name"]] = item.get("value", item.get("valueFrom"))
    return out


# ---------------------------------------------------------------------------
# pvc+hf -- the default
# ---------------------------------------------------------------------------


def test_pvc_hf_hands_the_chart_a_hub_cache_uri(tmp_path):
    """The uri carries the claim, the cache directory and the model id.

    The modelservice chart reads the id off the last two segments and the cache
    directory off everything between -- which is the same split
    ``_resolve_model_hub_cache`` does, so the job that writes the cache and the
    chart that reads it cannot drift.
    """
    _, plan_dir, merged = _render(tmp_path)

    artifacts = _ms_values(plan_dir)["modelArtifacts"]
    assert artifacts["uri"] == "pvc+hf://model-pvc/models/facebook/opt-125m"
    assert artifacts["name"] == "facebook/opt-125m"
    assert merged["model"]["hubCacheSubdir"] == "models"


def test_pvc_hf_mount_is_writable(tmp_path):
    """A hub cache is not a read-only artifact: the Hub client writes locks and
    resolves refs under the cache root, so mounting it read-only fails the very
    first lookup. Upstream's own values.yaml says as much for this protocol."""
    _, plan_dir, _ = _render(tmp_path)

    assert _ms_values(plan_dir)["modelArtifacts"]["readOnly"] is False


def test_pvc_hf_serves_a_bare_id_against_the_staged_cache(tmp_path):
    """The command names the model, not a path, and ``HF_HUB_CACHE`` is what
    makes that id resolve locally.

    We do not set that variable: the chart derives it from the uri we hand it, so
    setting it too would be a second, duplicate env entry that could drift. What
    this checks is that the chart's derivation lands on the directory the rest of
    the plan uses -- and that the container is mounting the volume at all, since
    the chart gates the whole derivation on that.
    """
    _, plan_dir, merged = _render(tmp_path)

    command = merged["decode"]["engine"]["command"]
    assert "facebook/opt-125m" in command
    assert "/model-cache" not in command

    values = _ms_values(plan_dir)
    container = values["decode"]["containers"][0]
    assert container["mountModelVolume"] is True

    env = _env(container)
    assert "HF_HUB_CACHE" not in env
    artifacts = values["modelArtifacts"]
    assert _chart_hub_cache_subdir(artifacts) == merged["model"]["hubCacheSubdir"]
    # And the chart mounts the volume where it says it does, so the directory the
    # engine ends up reading is the mount path plus that subdirectory.
    assert artifacts["mountPath"] == "/model-cache"

    # HF_HOME stays writable scratch: HF_HUB_CACHE outranks it for hub lookups,
    # so the two coexist rather than competing.
    assert env["HF_HOME"] == "/tmp/huggingface"


def test_pvc_hf_download_job_writes_the_directory_the_pods_read(tmp_path):
    """One subdirectory of one volume, seen from two different mount paths.

    The job populates the cache with a plain ``hf download <id>`` into
    ``HF_HUB_CACHE``, which is the layout a bare id resolves against -- not the
    flat ``--local-dir`` snapshot. It mounts the volume at its own path, so the
    absolute directory it writes is NOT the one the serving pods read; what has
    to match is the subdirectory inside the volume, because that is what both
    sides compose with their own mount.
    """
    _, plan_dir, merged = _render(tmp_path)

    job = yaml.safe_load((plan_dir / "04_download_job.yaml").read_text())
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = _env(container)

    mount = container["volumeMounts"][0]["mountPath"]
    subdir = merged["model"]["hubCacheSubdir"]
    assert env["HF_HUB_CACHE"] == f"{mount}/{subdir}"
    # Written through the same claim the serving pods mount, and into the same
    # place inside it as the chart derives from the uri they get.
    volume = job["spec"]["template"]["spec"]["volumes"][0]
    artifacts = _ms_values(plan_dir)["modelArtifacts"]
    assert volume["persistentVolumeClaim"]["claimName"] in artifacts["uri"]
    assert _chart_hub_cache_subdir(artifacts) == subdir

    assert env["HF_MODEL_ID"] == merged["model"]["name"]
    args = "\n".join(container["args"])
    assert 'hf download "${HF_MODEL_ID}"' in args
    assert "--local-dir" not in args


# ---------------------------------------------------------------------------
# hf -- no PVC at all
# ---------------------------------------------------------------------------


def test_hf_needs_no_pvc_and_serves_the_same_command(tmp_path):
    """Switching to ``hf`` drops the staging step; the engine command is
    untouched, because a bare id is what the Hub wants too.

    Nothing is derived either: there is no cache directory of ours to point at,
    and the chart sets ``HF_HOME`` to the mount itself on this protocol, so the
    values file leaves that variable alone rather than shadowing it.
    """
    _, plan_dir, merged = _render(tmp_path, {"modelservice": {"uriProtocol": "hf"}})

    values = _ms_values(plan_dir)
    assert values["modelArtifacts"]["uri"] == "hf://facebook/opt-125m"
    assert merged["model"]["hubCacheSubdir"] is None
    assert "facebook/opt-125m" in merged["decode"]["engine"]["command"]
    assert "HF_HOME" not in _env(values["decode"]["containers"][0])

    # Weights come straight from the Hub, so standup stages nothing first.
    step = ModelNamespaceStep()
    assert step._requires_pvc_download(merged) is False


# ---------------------------------------------------------------------------
# pvc -- a raw weights directory
# ---------------------------------------------------------------------------


def test_pvc_points_at_a_raw_directory_and_derives_no_cache(tmp_path):
    """``pvc`` is the escape hatch for a volume that already holds weights laid
    out flat. No cache layout applies, so nothing is derived and no
    ``HF_HUB_CACHE`` is set -- a command on this protocol names the directory
    itself."""
    _, plan_dir, merged = _render(tmp_path, {"modelservice": {"uriProtocol": "pvc"}})

    values = _ms_values(plan_dir)
    assert values["modelArtifacts"]["uri"] == "pvc://model-pvc/models/facebook/opt-125m"
    assert merged["model"]["hubCacheSubdir"] is None
    assert "HF_HUB_CACHE" not in _env(values["decode"]["containers"][0])


# ---------------------------------------------------------------------------
# the one shape pvc+hf cannot serve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["weights", "facebook/opt-125m"])
def test_a_path_too_short_for_a_hub_cache_is_an_error(tmp_path, path):
    """``pvc+hf`` spends the last two segments on the model id, so a path with
    nothing left over has no cache directory. Rendering it anyway would give a
    green pod serving a nonsense id against an empty cache, so it fails with the
    fix in the message."""
    result, _, _ = _render(tmp_path, {"model": {"path": path}}, expect_errors=True)

    errors = [
        msg
        for stack in result.stacks.values()
        for msg in (stack.render_errors or [])
        if "uriProtocol" in msg
    ]
    assert errors, "a too-short model.path must be reported"
    assert "pvc+hf" in errors[0]
    assert "models/<org>/<model>" in errors[0]
