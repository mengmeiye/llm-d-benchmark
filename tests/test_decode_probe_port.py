"""Tests for decode probe port rendering."""

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


def _render_pd_disaggregation(tmp_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    result = RenderPlans(
        template_dir=root / "config" / "templates" / "jinja",
        defaults_file=root / "config" / "templates" / "values" / "defaults.yaml",
        scenarios_file=(
            root / "config" / "scenarios" / "guides" / "pd-disaggregation.yaml"
        ),
        output_dir=tmp_path / "plan",
        logger=_Logger(),
    ).eval()

    assert not result.has_errors
    values_path = tmp_path / "plan" / "pd-disaggregation" / "13_ms-values.yaml"
    return yaml.safe_load(values_path.read_text(encoding="utf-8"))


def _render_plan_values(tmp_path: Path) -> dict[str, Any]:
    """The merged/resolved config the templates were rendered from."""
    path = tmp_path / "plan" / "pd-disaggregation" / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_decode_probes_follow_the_port_in_the_engine_command(
    tmp_path: Path,
) -> None:
    """Every probe must aim at the port decode's command actually binds.

    The guide's decode command says ``--port 8200`` and nothing else in the
    plan states a port: the routing sidecar holds 8000 on that pod, so a probe
    pointed there would pass while the engine was still loading -- or keep
    passing after it died. The number is read out of the command once
    in the resolved decode snapshot and every consumer follows it, so changing the
    command is the only edit a user has to make.
    """
    values = _render_pd_disaggregation(tmp_path)

    decode_container = values["decode"]["containers"][0]
    extra_config = decode_container["extraConfig"]

    # The command is passed to `/bin/bash -c` as the last arg, verbatim.
    assert "--port 8200" in decode_container["args"][-1], (
        "the command must render verbatim, port included"
    )

    for probe, field in (
        ("startupProbe", "httpGet"),
        ("livenessProbe", "tcpSocket"),
        ("readinessProbe", "httpGet"),
    ):
        assert extra_config[probe][field]["port"] == 8200, (
            f"{probe} should probe the engine's own port, not the sidecar's"
        )


def test_decode_engine_port_is_read_from_the_command(tmp_path: Path) -> None:
    """``resolve_engines`` records the command's port for the rest of the plan.

    The sidecar's upstream, the container port and the PodMonitor all read
    the resolved decode port rather than re-parsing the command, so it has to
    be populated even though the scenario never writes it.
    """
    _render_pd_disaggregation(tmp_path)
    plan = _render_plan_values(tmp_path)

    assert plan["decode"]["engine"]["port"] == 8200
    assert plan["resolvedServingRoles"]["decode"]["port"] == 8200
    # Prefill gets no sidecar, so its command binds the Service port itself.
    assert plan["prefill"]["engine"]["port"] == 8000
    assert plan["engine"]["servicePort"] == 8000

    # The device count is not read from the command: `parallelism.tensor` is a
    # chart value the scenario states, because the kubelet grants accelerators
    # before the engine process exists. Decode says 2 (matching its
    # `--tensor-parallel-size 2`); prefill says nothing and takes the default.
    assert plan["decode"]["parallelism"]["tensor"] == 2
    assert plan["prefill"]["parallelism"]["tensor"] == 1
    assert "acceleratorCount" not in plan["decode"]["engine"]


def test_capacity_metrics_come_from_the_command(tmp_path: Path) -> None:
    """The scenario states none of these numbers; the command does.

    Three are read back, each because something outside the engine has to agree
    with it: the pre-deploy capacity check sizes KV cache against the context
    length and the memory fraction (the harness workload profile takes its
    context length from the same pair), and a prefix-cache router hashes on the
    KV page size. Reading them is what lets the scenario state each one once, in
    the flag the engine actually reads.
    """
    _render_pd_disaggregation(tmp_path)
    plan = _render_plan_values(tmp_path)

    assert plan["model"]["maxModelLen"] == 16384
    assert plan["model"]["gpuMemoryUtilization"] == 0.90
    assert plan["model"]["blockSize"] == 128

    # Nothing else is. `--max-num-seqs 256` is in the command, read by vLLM and
    # by no one else -- there is no key at all for the batch widths.
    assert set(plan["model"]) & {"maxNumSeq", "maxNumBatchedTokens"} == set()
    assert set(plan["model"]) & {"maxNumSeq", "maxNumBatchedTokens"} == set()
