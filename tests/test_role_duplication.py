"""Role blocks in one stack must not restate the same value.

``decode`` and ``prefill`` are usually the same pod, and before ``roleDefaults:``
existed a scenario that configured both said everything twice. The copies drifted
-- one role got a probe timeout the other did not, one got ``shm`` -- and nothing
caught it, because a restatement is not a syntax error.

``roleDefaults:`` states the shared shape once, so this file holds the line: a
key two roles in the same stack state identically is either moved into
``roleDefaults`` or declared in :data:`KEPT` with the reason it stays. ``enabled``
is exempt everywhere -- which roles a stack deploys is the one thing worth
reading off each role block directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "config" / "scenarios"

#: Roles one ``roleDefaults:`` block may seed. ``nok8s`` is not among them: it
#: describes an ssh connection, not a pod.
ROLES = ("decode", "prefill", "standalone")

#: Restatements that stay, and why. Every entry is a judgement, not an oversight,
#: and the list should only ever shrink.
KEPT: dict[tuple[str, str], str] = {
    ("examples/eval-containers-aider-polyglot.yaml", "accelerator"): (
        "defaults.yaml models `accelerator` for no role, so roleDefaults "
        "rejects it as a key no role defines -- while config_schema's "
        "DeploymentBaseConfig declares it and ten scenarios set it. The "
        "inconsistency is in defaults.yaml, and fixing it there is a separate "
        "change from de-duplicating scenarios."
    ),
    ("examples/eval-containers-gaia.yaml", "accelerator"): "See above.",
    ("examples/eval-containers-aider-polyglot.yaml", "acceleratorType"): (
        "decode and prefill agree, but defaults.yaml models "
        "`standalone.acceleratorType` too, so hoisting would newly seed a role "
        "that never asked for it. A de-duplication must not change what a role "
        "resolves to."
    ),
    ("cicd/kind.yaml", "parallelism"): (
        "Same shape as acceleratorType above: standalone models `parallelism` "
        "and does not state it, so hoisting decode's `tensor: 0` would silently "
        "give the standalone alternative a different width."
    ),
    ("examples/eval-containers-aider-polyglot.yaml", "engine"): (
        "The values coincide only because the sim image takes no command "
        '(`command: ""`). Each role\'s block explains which port that leaves it '
        "on and why, and that prose is genuinely per-role -- hoisting the key "
        "would have to delete two of the three explanations."
    ),
    ("examples/eval-containers-gaia.yaml", "engine"): "See above.",
    ("cicd/kind.yaml", "engine"): "See above.",
    ("examples/sim.yaml", "engine"): "See above.",
    ("guides/wide-ep.yaml", "replicas"): (
        "How many replicas a role runs is the per-role knob, and prefill keeps a "
        "commented-out `replicas: 8` alternative beside it that would mean "
        "something else under roleDefaults."
    ),
    ("cicd/cks.yaml", "replicas"): (
        "One line stated twice in a stack with no other shared key: a "
        "roleDefaults block would cost more lines than it saves."
    ),
    ("cicd/gke.yaml", "replicas"): "See above.",
    ("cicd/ocp.yaml", "replicas"): "See above.",
    ("guides/keda-epp-token-aware-pd-disaggregation.yaml", "replicas"): (
        "The two 1s mean different things: prefill runs one pod, and decode's 1 "
        "is the eppKedaSaturation minReplicaCount floor the autoscaler scales up "
        "from. Each role's comment says which, and hoisting would state a "
        "starting replica count for an autoscaled role and a fixed one as if "
        "they were the same decision."
    ),
}


def _layers(doc):
    """Every mapping a stack may write role blocks in: the stack, and
    ``modelservice:`` -- both spellings are in use."""
    for stack in (doc or {}).get("scenario") or []:
        if not isinstance(stack, dict):
            continue
        yield stack
        nested = stack.get("modelservice")
        if isinstance(nested, dict):
            yield nested


def _restatements(path: Path) -> set[str]:
    """Keys two or more roles in one layer of this scenario state identically."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for layer in _layers(doc):
        roles = {r: layer[r] for r in ROLES if isinstance(layer.get(r), dict)}
        if len(roles) < 2:
            continue
        for key in set().union(*(set(block) for block in roles.values())):
            if key == "enabled":
                continue
            holders = [r for r, block in roles.items() if key in block]
            if len(holders) < 2:
                continue
            if len({yaml.dump(roles[r][key]) for r in holders}) == 1:
                found.add(key)
    return found


@pytest.fixture(scope="module")
def observed() -> dict[tuple[str, str], None]:
    files = sorted(SCENARIOS.rglob("*.yaml"))
    # Guard the guard: an empty corpus would make the assertions below pass.
    assert len(files) > 30, "scenario corpus looks truncated"
    return {
        (str(f.relative_to(SCENARIOS)), key): None
        for f in files
        for key in _restatements(f)
    }


def test_no_undeclared_restatement(observed):
    """The rule: hoist it into ``roleDefaults``, or say why it stays."""
    extra = sorted(
        f"{name}: {key}" for name, key in observed if (name, key) not in KEPT
    )
    assert extra == [], (
        "these role keys are stated identically by two roles in the same "
        "stack:\n  "
        + "\n  ".join(extra)
        + "\n\nMove each one into that stack's `roleDefaults:` block (see "
        "config/README.md), which every role inherits and any role can "
        "override. If it genuinely has to stay, add it to KEPT with the reason."
    )


def test_declared_restatements_still_exist(observed):
    """A stale exemption is an exemption with nothing behind it.

    Left in place, it is where the next restatement of that key hides.
    """
    gone = sorted(f"{name}: {key}" for name, key in KEPT if (name, key) not in observed)
    assert gone == [], (
        "these KEPT entries no longer describe anything:\n  "
        + "\n  ".join(gone)
        + "\n\nThe duplication is gone -- delete the entry."
    )


def test_kept_list_only_shrinks():
    """A ratchet. Raising this number means shipping another restatement."""
    assert len(KEPT) <= 13
