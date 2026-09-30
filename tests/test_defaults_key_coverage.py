"""Every key in ``defaults.yaml`` must reach something that reads it.

``defaults.yaml`` is the configuration surface: a user scans it to learn what
they are allowed to set. A key nothing consumes is worse than clutter, because
it reads as a promise -- someone sets it, nothing happens, and the deployment is
silently not what they asked for. ``idleCleanup`` described a CronJob that had
no template; ``fma.requester.readinessProbeInitialDelay`` named a probe field
the manifest hardcoded; ``experiment.analyzeLocally`` shadowed a flag that is
really spelled ``--analyze``. None of them were visible in review.

So this test reads the file the way a user does -- key by key -- and requires
each leaf to be reachable from code or a template. Three things are reachable
without being named, and each has to be declared here rather than inferred:

``_ANCHOR_SUBTREE``
    YAML anchors, resolved by the parser inside this very file. The name never
    appears anywhere else because ``*gpu_label_key`` is the reference.

``WHOLESALE_SUBTREES``
    Subtrees a template dumps entire -- ``{{ router | toyaml }}``,
    ``{% for k, v in prism.env.items() %}``. The leaves reach the manifest
    without ever being named, so a leaf-name grep cannot see them. Adding a key
    under one of these is free and needs no code change, which is the point.

``UNWIRED``
    Keys that are genuinely not consumed and are **bugs**, recorded with what
    the fix would be. This list should only ever shrink. It exists so the rest
    of the file can be held to the rule while the open questions stay visible.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ROOT / "config" / "templates" / "values" / "defaults.yaml"

#: Anchors are dereferenced by the YAML parser, in this file, by ``*name``.
_ANCHOR_SUBTREE = ("_anchors",)

#: Paths whose whole subtree is rendered by one expression. Each entry names
#: the template and the expression, so the claim can be rechecked.
WHOLESALE_SUBTREES: dict[tuple[str, ...], str] = {
    ("router",): "12_router-values.yaml.j2: {{ router | toyaml }}",
    ("prism", "env"): "35_prism.yaml.j2: {% for key, value in prism.env.items() %}",
    ("fma", "labels"): "24_fma-deployment.yaml.j2: fma.labels.items()",
    (
        "wva",
        "hpa",
        "behavior",
    ): "28_wva-scaledobject.yaml.j2: {{ wva.hpa.behavior | toyaml }}",
    (
        "eppKedaSaturation",
        "scaledObject",
        "behavior",
    ): "30_keda-scaledobject.yaml.j2: {{ ...behavior | toyaml }}",
}

#: Keys no consumer reads. Every one of these is a bug; the note says which.
UNWIRED: dict[tuple[str, ...], str] = {
    ("wva", "namespaceScoped"): (
        "Set by config/scenarios/guides/workload-autoscaling.yaml and described "
        "in docs/workload-variant-autoscaler.md, but no template branches on "
        "it, so a namespace-scoped WVA install renders cluster-scoped."
    ),
    ("eppKedaSaturation", "epp", "prometheusServiceAccount"): (
        "Four scenarios set this to thanos-querier, but "
        "29_epp-keda-saturation-epp-monitoring.yaml.j2 names "
        "prometheus-user-workload literally. The two are different service "
        "accounts with different jobs (KEDA's query frontend vs. the scraper "
        "that reads the metrics secret), so which one the RoleBinding should "
        "name needs checking on a cluster before either is changed."
    ),
    ("eppKedaSaturation", "epp", "prometheusServiceAccountNamespace"): (
        "Same RoleBinding as prometheusServiceAccount above; the template names "
        "openshift-user-workload-monitoring literally."
    ),
}

#: File types that can read a plan value. Scenarios and docs are deliberately
#: excluded: a key a scenario sets and nothing reads is the bug, not the proof.
CONSUMER_SUFFIXES = {".py", ".j2", ".sh", ".tpl"}


def _leaf_paths(node, path: tuple[str, ...] = ()):
    if isinstance(node, dict) and node:
        for key, value in node.items():
            yield from _leaf_paths(value, path + (str(key),))
    else:
        yield path


def _consumer_text() -> str:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    parts: list[str] = []
    for name in tracked:
        path = ROOT / name
        if not path.is_file() or name.startswith("tests/"):
            continue
        if path.suffix not in CONSUMER_SUFFIXES:
            continue
        try:
            parts.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return "\n".join(parts)


@pytest.fixture(scope="module")
def consumers() -> str:
    text = _consumer_text()
    # Guard the guard: an empty corpus would make every assertion below pass.
    assert len(text) > 500_000, "consumer corpus looks truncated"
    return text


@pytest.fixture(scope="module")
def leaf_paths() -> list[tuple[str, ...]]:
    paths = list(_leaf_paths(yaml.safe_load(DEFAULTS.read_text(encoding="utf-8"))))
    assert len(paths) > 500, "defaults.yaml looks truncated"
    return paths


def _covered_by(path: tuple[str, ...], subtrees) -> tuple[str, ...] | None:
    for prefix in subtrees:
        if path[: len(prefix)] == prefix:
            return prefix
    return None


def test_every_key_reaches_a_consumer(consumers, leaf_paths):
    """The rule: a key is named by code or a template, or it is declared above."""
    orphans: list[str] = []

    for path in leaf_paths:
        if _covered_by(path, [_ANCHOR_SUBTREE]):
            continue
        if _covered_by(path, WHOLESALE_SUBTREES):
            continue
        if path in UNWIRED:
            continue
        leaf = path[-1]
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(leaf)}(?![A-Za-z0-9_])", consumers):
            continue
        orphans.append(".".join(path))

    assert orphans == [], (
        "these defaults.yaml keys are not read by any .py/.j2/.sh:\n  "
        + "\n  ".join(orphans)
        + "\n\nDelete the key, wire it up, or -- if it is reached through a "
        "parent that a template dumps entire -- add that parent to "
        "WHOLESALE_SUBTREES with the expression that dumps it."
    )


def test_wholesale_subtrees_still_exist(leaf_paths):
    """A declared exemption that matches nothing is stale; drop it.

    Otherwise the exemption outlives the template that justified it and starts
    hiding orphans under a path nobody dumps any more.
    """
    for prefix in list(WHOLESALE_SUBTREES) + [_ANCHOR_SUBTREE]:
        assert any(p[: len(prefix)] == prefix for p in leaf_paths), (
            f"{'.'.join(prefix)} is exempted but no longer exists in "
            f"defaults.yaml -- remove the entry"
        )


@pytest.mark.parametrize("prefix,where", sorted(WHOLESALE_SUBTREES.items()))
def test_wholesale_subtrees_are_actually_dumped(prefix, where):
    """The exemption has to be true: some template dumps this path entire."""
    template = ROOT / "config" / "templates" / "jinja" / where.split(":", 1)[0]
    assert template.is_file(), f"{where}: no such template"

    text = template.read_text(encoding="utf-8")
    expr = ".".join(prefix)
    dumped = re.search(
        rf"{re.escape(expr)}\s*(?:\|\s*to(?:yaml|json)|\.items\(\))", text
    )
    assert dumped, (
        f"{where} no longer dumps {expr} wholesale, so its leaves are not "
        f"reaching the manifest. Either fix the template or stop exempting it."
    )


def test_unwired_keys_are_still_present(leaf_paths):
    """When a bug above is fixed, its entry must go too.

    A stale UNWIRED entry is an exemption with nothing behind it, which is how
    the next orphan gets in unnoticed.
    """
    present = set(leaf_paths)
    for path in UNWIRED:
        assert path in present, (
            f"{'.'.join(path)} is listed in UNWIRED but no longer exists in "
            f"defaults.yaml -- remove the entry"
        )


def test_unwired_list_only_shrinks():
    """A ratchet. Raising this number means shipping another key that lies."""
    assert len(UNWIRED) <= 3


def test_deleted_sections_stay_deleted(leaf_paths):
    """Named so a revert is a test failure rather than a silent restoration."""
    tops = {p[0] for p in leaf_paths}

    assert "idleCleanup" not in tops, (
        "idleCleanup describes a CronJob with no template. If it is coming "
        "back, the template comes with it."
    )
    assert "openshiftMonitoring" not in tops, (
        "openshiftMonitoring held one namespace that its would-be consumer "
        "names literally -- correctly, since OpenShift fixes it."
    )
