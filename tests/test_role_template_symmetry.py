"""decode and prefill are one description, not two.

The two engine roles are the same pod -- same container, same probes, same
volumes, same accelerator arithmetic -- differing only in the values they are
given. They used to be written out twice, 440 lines each, and the copies drifted
exactly where you would expect: prefill's podmonitor tested ``is defined`` where
decode tested the value, and prefill's container was named ``vllm`` in a chart
that is meant to run whichever engine the user's command launches. Neither
divergence was visible in review, because reviewing it meant diffing two blocks
400 lines apart.

Both roles now render through one ``modelservice_role`` macro, so the drift is
structurally impossible rather than merely discouraged. These tests keep it that
way: they fail if a role-specific block grows back in the template, and they fail
if an engine's name is written into the shared macro.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
JINJA = ROOT / "config" / "templates" / "jinja"
MS_VALUES = JINJA / "13_ms-values.yaml.j2"
MACROS = JINJA / "_macros.j2"

ROLES = ("decode", "prefill")

#: Engines llm-d-benchmark can stand up. None of their names belongs in a
#: template shared by every engine.
ENGINE_NAMES = ("vllm", "sglang", "trtllm", "tensorrt")

#: Role references outside the role blocks that are not duplication. A role name
#: is legitimate here when the field it configures genuinely belongs to one role.
#: Each entry has to say why, so an accidental one cannot hide among them.
ALLOWED_ROLE_REFERENCES: dict[str, str] = {
    "targetPort: {{ routing.proxy.targetPort | default(resolvedServingRoles.decode.port, true) }}": (
        "The routing sidecar is injected into decode pods only -- the chart "
        "never puts it on prefill -- so the proxy's target is the resolved "
        "decode port by definition. This is a top-level `routing:` field, not "
        "a copy of a role block."
    ),
}


def _strip_comments(text: str) -> str:
    """Jinja comments only document; they cannot render a field."""
    return re.sub(r"\{#.*?#\}", "", text, flags=re.S)


@pytest.fixture(scope="module")
def ms_values() -> str:
    text = MS_VALUES.read_text(encoding="utf-8")
    assert "modelservice_role" in text, "guard the guard: wrong template"
    return text


@pytest.fixture(scope="module")
def role_macro() -> str:
    """The body of ``modelservice_role``, comments stripped."""
    text = MACROS.read_text(encoding="utf-8")
    start = text.index("{% macro modelservice_role(")
    body = text[start : text.index("{% endmacro %}", start)]
    # Guard the guard: a mis-sliced body would make the assertions vacuous.
    assert body.count("\n") > 300, (
        f"macro body looks truncated: {body.count(chr(10))} lines"
    )
    return _strip_comments(body)


def test_exactly_one_role_macro() -> None:
    """Two definitions would be the duplication back under another name."""
    text = MACROS.read_text(encoding="utf-8")
    assert text.count("{% macro modelservice_role(") == 1


@pytest.mark.parametrize("role", ROLES)
def test_each_role_is_rendered_by_the_shared_macro(ms_values: str, role: str) -> None:
    assert f"modelservice_role('{role}', {role})" in ms_values, (
        f"{role} is no longer rendered through modelservice_role. Both roles "
        f"must go through the one macro, or they will drift again."
    )


@pytest.mark.parametrize("role", ROLES)
def test_no_role_specific_field_rendering_survives(ms_values: str, role: str) -> None:
    """The template may name a role only to pass it to the macro.

    Any other ``decode.``/``prefill.`` reference is a field being rendered for
    one role, which is how the second copy starts.
    """
    body = _strip_comments(ms_values)
    offenders = [
        line.strip()
        for line in body.split("\n")
        if re.search(rf"\b{role}[._]", line)
        and "modelservice_role(" not in line
        and line.strip() not in ALLOWED_ROLE_REFERENCES
    ]
    assert offenders == [], (
        f"these lines render fields for {role} outside the shared macro:\n  "
        + "\n  ".join(offenders)
        + "\n\nMove the field into modelservice_role, or -- if it really "
        "belongs to one role -- add it to ALLOWED_ROLE_REFERENCES with the "
        "reason."
    )


def test_allowed_role_references_are_still_there(ms_values: str) -> None:
    """A stale allowance is an exemption with nothing behind it."""
    body = _strip_comments(ms_values)
    lines = {line.strip() for line in body.split("\n")}
    for allowed in ALLOWED_ROLE_REFERENCES:
        assert allowed in lines, (
            f"this line is allowed to name a role but no longer exists:\n  "
            f"{allowed}\nRemove the ALLOWED_ROLE_REFERENCES entry."
        )


@pytest.mark.parametrize("engine", ENGINE_NAMES)
def test_shared_macro_names_no_engine(role_macro: str, engine: str) -> None:
    """``- name: "vllm"`` on the prefill container was this bug.

    The serving container is named from ``engine.containerName``, which defaults
    to llm-d's engine-neutral ``modelserver`` -- so ``kubectl logs -c
    modelserver`` reads the same whichever engine the role's command launches.
    """
    assert engine not in role_macro.lower(), (
        f"the shared role macro mentions {engine!r}. It renders every engine; "
        f"read the value from the plan instead of writing an engine's name in."
    )


def test_macro_takes_only_the_role_and_its_values() -> None:
    """A per-role parameter is a divergence with a signature.

    ``container_name`` was one: it existed so prefill could keep passing
    ``'vllm'``. Each extra parameter is somewhere the two roles are still
    described differently, so the signature is held to role name + values.
    """
    text = MACROS.read_text(encoding="utf-8")
    signature = re.search(r"\{% macro modelservice_role\(([^)]*)\)", text)
    assert signature, "modelservice_role signature not found"
    params = [p.strip() for p in signature.group(1).split(",") if p.strip()]
    assert params == ["role_name", "role"], (
        f"unexpected parameters {params}. A new one usually means one role "
        f"needs a value the other does not -- read it from `role` instead."
    )
