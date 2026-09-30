"""Every registered validator must be reachable, and every stack resolvable.

``get_validator()`` falls back to ``BaseSmoketest`` for an unknown stack name.
That fallback is the right behaviour -- most scenarios have no dedicated
validator and should still get health checks -- but it makes one mistake
invisible: a validator registered under a name no scenario declares never runs,
and the smoketest passes having checked nothing scenario-specific.

That is not hypothetical. Three validators were wired to names nothing deployed:
``inference-scheduling-wva`` (the stack was renamed ``workload-autoscaling``),
``precise-prefix-cache-aware`` (the stack is ``precise-prefix-cache-routing``)
and ``cpu-example-ms`` (the stack is ``cpu-example``). Each looked fine in review
because the registry and the scenario were never read side by side.

These tests read them side by side.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llmdbenchmark.smoketests import get_validator
from llmdbenchmark.smoketests.base import BaseSmoketest
from llmdbenchmark.smoketests.validators import (
    VALIDATORS,
    _SYNTHETIC_STACK_NAMES,
)


SCENARIOS = Path(__file__).resolve().parents[1] / "config" / "scenarios"


def _stack_names() -> set[str]:
    """Every ``scenario[].name`` across every scenario file."""
    names: set[str] = set()
    for path in SCENARIOS.rglob("*.yaml"):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (
            yaml.YAMLError
        ) as exc:  # pragma: no cover - a parse error is its own test
            pytest.fail(f"{path} is not parseable YAML: {exc}")
        if not isinstance(doc, dict):
            continue
        for stack in doc.get("scenario") or []:
            if isinstance(stack, dict) and stack.get("name"):
                names.add(str(stack["name"]))
    return names


def test_scenarios_declare_stack_names() -> None:
    """Guard the guard: an empty set would make every test below vacuous."""
    assert len(_stack_names()) > 20


@pytest.mark.parametrize("key", sorted(VALIDATORS))
def test_every_registered_validator_matches_a_stack(key: str) -> None:
    """A registry key is a stack name, not a filename and not a guess.

    If this fails, either the scenario was renamed and the registry was not, or
    the key is deliberately synthetic and belongs in
    ``_SYNTHETIC_STACK_NAMES`` with a comment saying why.
    """
    if key in _SYNTHETIC_STACK_NAMES:
        pytest.skip(f"{key} is a documented synthetic stack name")

    assert key in _stack_names(), (
        f"VALIDATORS['{key}'] -> {VALIDATORS[key].__name__} matches no "
        f"scenario's stack name, so it never runs and the smoketest silently "
        f"falls back to BaseSmoketest. Fix the key, rename the stack, or add it "
        f"to _SYNTHETIC_STACK_NAMES."
    )


def test_synthetic_names_are_not_also_real_stacks() -> None:
    """A name in both places means the allowlist is hiding a live mapping."""
    overlap = _SYNTHETIC_STACK_NAMES & _stack_names()
    assert overlap == set(), (
        f"{sorted(overlap)} are declared synthetic but a scenario deploys them; "
        f"remove them from _SYNTHETIC_STACK_NAMES so the check above applies."
    )


def test_the_renamed_wva_stack_resolves_to_its_validator() -> None:
    """The rename's whole point: these checks used to be skipped silently."""
    validator = get_validator("workload-autoscaling")

    assert type(validator) is not BaseSmoketest
    # OptimizedBaselineValidator mixes in the WVA checks, which activate on
    # wva.enabled -- that is what a WVA scenario needs and was not getting.
    assert hasattr(validator, "validate_wva_resources") or any(
        "Wva" in base.__name__ for base in type(validator).__mro__
    )


@pytest.mark.parametrize(
    "stack,expected",
    [
        ("precise-prefix-cache-routing", "PrecisePrefixCacheAwareValidator"),
        ("cpu-example", "CpuValidator"),
        ("workload-autoscaling", "OptimizedBaselineValidator"),
    ],
)
def test_previously_unreachable_validators_now_resolve(
    stack: str, expected: str
) -> None:
    assert type(get_validator(stack)).__name__ == expected


def test_unknown_stack_still_falls_back() -> None:
    """The fallback is intended behaviour for the scenarios with no validator."""
    assert type(get_validator("no-such-stack")) is BaseSmoketest
