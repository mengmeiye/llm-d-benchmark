"""UCX_NET_DEVICES reaches the serving containers, and stays out by default.

Pinning UCX to a device only matters once a pod has more than one interface,
which is exactly the case a ``k8s.v1.cni.cncf.io/networks`` annotation creates.
The knob must therefore be renderable, and must stay absent from stacks that do
not transfer KV blocks so that single-interface pods keep letting UCX
auto-select.

UCX reads its settings from the container environment, so the knob is an
``extraEnvVars`` entry -- stated once in a stack's ``roleDefaults`` and
inherited by every role -- rather than a template-rendered key. Nothing
translates it: what a scenario writes is what the container gets. These tests
render the real guides and check that.

``pd-disaggregation.yaml`` is the RDMA stack: it ships ``UCX_TLS`` with ``rc``
and ``UCX_NET_DEVICES`` empty, so device discovery in the preprocess init
container picks the rails. ``wide-ep.yaml`` carries the absence case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llmdbenchmark.parser.render_plans import RenderPlans


class _Logger:
    def log_info(self, *_: Any, **__: Any) -> None:
        pass

    def log_warning(self, *_: Any, **__: Any) -> None:
        pass

    def log_error(self, *_: Any, **__: Any) -> None:
        pass

    def log_debug(self, *_: Any, **__: Any) -> None:
        pass

    def line_break(self) -> None:
        pass


def _render(
    tmp_path: Path,
    overrides: dict | None = None,
    guide: str = "wide-ep",
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    result = RenderPlans(
        template_dir=root / "config" / "templates" / "jinja",
        defaults_file=root / "config" / "templates" / "values" / "defaults.yaml",
        scenarios_file=(root / "config" / "scenarios" / "guides" / f"{guide}.yaml"),
        output_dir=tmp_path / "plan",
        logger=_Logger(),
        setup_overrides=overrides,
    ).eval()

    assert not result.has_errors
    values_path = tmp_path / "plan" / guide / "13_ms-values.yaml"
    return yaml.safe_load(values_path.read_text(encoding="utf-8"))


def _env(values: dict[str, Any], role: str) -> dict[str, str]:
    container = values[role]["containers"][0]
    return {item["name"]: item["value"] for item in container["env"] if "value" in item}


def test_ucx_is_absent_from_a_stack_that_does_not_ask_for_it(tmp_path: Path) -> None:
    """wide-ep transfers no KV blocks between pods, so it sets no UCX knob.

    The guide documents both variables in a comment on its ``engine:`` block.
    A comment is all it should be: nothing renders a UCX default onto a role
    that never asked, which is what lets UCX auto-select on the single
    interface these pods have.
    """
    values = _render(tmp_path)

    for role in ("prefill", "decode"):
        env = _env(values, role)
        # Other extraEnvVars entries do render, so absence here is this stack
        # not setting UCX rather than the env list being dropped.
        assert env, f"{role} rendered no environment at all"
        assert "UCX_TLS" not in env
        assert "UCX_NET_DEVICES" not in env


def test_pd_disaggregation_enables_rdma_transport(tmp_path: Path) -> None:
    """The PD guide selects the RDMA transport and leaves the rail pin open.

    ``rc`` has to stay in UCX_TLS. The guide ships UCX_NET_DEVICES empty, which
    is what the preprocess init container reads as "discover the rails" -- it
    then writes the devices it found into the env file the command sources,
    which is why an empty value here is not the same as no value at all.
    """
    values = _render(tmp_path, guide="pd-disaggregation")

    for role in ("prefill", "decode"):
        env = _env(values, role)

        assert "rc" in env["UCX_TLS"].split(",")
        assert env["UCX_NET_DEVICES"] == ""


def test_pd_disaggregation_pin_reaches_both_roles_in_hca_form(
    tmp_path: Path,
) -> None:
    """An explicit pin reaches both roles of the PD guide unchanged.

    Each rail must stay in HCA form (``mlx5_1:1``) rather than netdev form
    (``eth0``), which is what the RDMA transport requires -- and the value must
    arrive byte-for-byte, because the preprocess only steps aside for a
    non-empty one.

    The pin is stated per role here rather than in ``roleDefaults``: role
    blocks are what ``roleDefaults`` seeds, and a CLI/setup override lands
    after that seeding.
    """
    pin = "mlx5_1:1,mlx5_3:1"
    entry = [{"name": "UCX_NET_DEVICES", "value": pin}]
    values = _render(
        tmp_path,
        {"prefill": {"extraEnvVars": entry}, "decode": {"extraEnvVars": entry}},
        guide="pd-disaggregation",
    )

    for role in ("prefill", "decode"):
        env = _env(values, role)

        assert env["UCX_NET_DEVICES"] == pin
        for rail in env["UCX_NET_DEVICES"].split(","):
            device, _, port = rail.partition(":")
            assert device.startswith("mlx5_"), f"{rail!r} is not an HCA device"
            assert port, f"{rail!r} is missing the :<port> suffix"


def _render_standalone(tmp_path: Path, overrides: dict | None = None) -> str:
    """Render the standalone deployment, which has its own template."""
    root = Path(__file__).resolve().parents[1]
    setup_overrides: dict[str, Any] = {
        "modelservice": {"enabled": False},
        "standalone": {"enabled": True},
    }
    if overrides:
        setup_overrides.update(overrides)
    result = RenderPlans(
        template_dir=root / "config" / "templates" / "jinja",
        defaults_file=root / "config" / "templates" / "values" / "defaults.yaml",
        scenarios_file=root / "config" / "scenarios" / "examples" / "gpu.yaml",
        output_dir=tmp_path / "plan",
        logger=_Logger(),
        setup_overrides=setup_overrides,
    ).eval()

    assert not result.has_errors
    deployment = result.rendered_paths[0] / "14_standalone-deployment_yaml.yaml"
    text = deployment.read_text(encoding="utf-8")
    # Parsing guards the hand-indented env entries: a wrong indent would still
    # read fine as text but break the document.
    yaml.safe_load(text)
    return text


def _standalone_env(text: str) -> dict[str, str]:
    deployment = yaml.safe_load(text)
    containers = deployment["spec"]["template"]["spec"]["containers"]
    return {
        item["name"]: item["value"]
        for container in containers
        for item in container.get("env", [])
        if "value" in item
    }


def test_standalone_omits_ucx_by_default(tmp_path: Path) -> None:
    """A standalone pod has one peer -- itself -- so it gets no UCX settings."""
    text = _render_standalone(tmp_path)

    assert "UCX_TLS" not in text
    assert "UCX_NET_DEVICES" not in text


def test_standalone_renders_ucx_net_devices_from_extra_env_vars(
    tmp_path: Path,
) -> None:
    """``standalone.extraEnvVars`` is the documented home, and it renders."""
    text = _render_standalone(
        tmp_path,
        {
            "standalone": {
                "enabled": True,
                "extraEnvVars": [{"name": "UCX_NET_DEVICES", "value": "mlx5_1:1"}],
            }
        },
    )

    assert _standalone_env(text)["UCX_NET_DEVICES"] == "mlx5_1:1"
