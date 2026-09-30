"""Tests for smoketest validation of per-replica engine values.

Replicas of one role share a single pod template, so a scenario that wants a
different value per replica writes a ``,,``-joined list in
``<role>.extraEnvVars`` and references the variable from its engine command::

    decode:
      replicas: 2
      contextLengthRanges: ["0-1000", "1000-2048"]
      extraEnvVars:
        - name: MY_MAX_LEN
          value: "2048,,32768"
      engine:
        command: |
          vllm serve $MODEL_NAME --port 8200 --max-model-len $MY_MAX_LEN

The whole list is what lands in the pod spec; ``set_llmdbench_environment.py``
splits it at container start and re-exports the entry matching the pod's LWS
index. Comparing the raw spec value against one scalar therefore always fails,
which is what ``assert_env_variant_list`` exists to avoid.

Per-replica variation is entirely the user's own: a ``,,`` list in an env var
they name, referenced from their command. llm-d-benchmark joins nothing, names
no engine parameter, and has nothing to keep in step with a new engine release.
These tests pin that mechanism, plus the two checks ``validate_role_pods``
makes on it:

1. ``,,`` lists are compared against the joined value, not a scalar.
2. The role's command reaches the container verbatim (``engine_command``).
3. A per-replica env var declared in ``extraEnvVars`` is asserted as the list.
4. No env check is derived from an engine flag -- one would drag back the
   per-engine bookkeeping the verbatim command exists to avoid.
"""

from __future__ import annotations

import sys
import types

import pytest


# Stub planner so we can import smoketest modules (see
# test_smoketest_inference.py for the same pattern + rationale).
def _stub_attr(name: str):
    """Any real attribute is a no-op callable; dunders must still fail.

    ``inspect.getsourcefile()`` does ``module.__file__.endswith(...)``, and
    pydantic's docstring extraction walks ``sys.modules`` to get there. A stub
    that answers ``__file__`` with a callable turns an unrelated later
    collection into ``AttributeError: 'function' object has no attribute
    'endswith'`` -- so the suite passes or fails on filename sort order.
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return lambda *a, **kw: None


if "planner" not in sys.modules:
    planner_stub = types.ModuleType("planner")
    capacity_stub = types.ModuleType("planner.capacity_planner")
    capacity_stub.__getattr__ = _stub_attr  # type: ignore[attr-defined]
    sys.modules["planner"] = planner_stub
    sys.modules["planner.capacity_planner"] = capacity_stub

from llmdbenchmark.smoketests.base import BaseSmoketest  # noqa: E402


# ---------------------------------------------------------------------------
# assert_env_variant_list
# ---------------------------------------------------------------------------


class TestAssertEnvVariantList:
    """The helper compares against the ,,-joined list the scenario declared.

    The variable names below are the scenario author's own -- nothing in
    llm-d-benchmark knows them, which is the point: the same mechanism varies a
    context length under vLLM, a ``--context-length`` under SGLang, or anything
    else, without the helper learning a single flag.
    """

    def test_matching_list_passes(self):
        env = {"MY_MAX_LEN": "2048,,2048"}
        result = BaseSmoketest.assert_env_variant_list(env, "MY_MAX_LEN", [2048, 2048])
        assert result.passed
        assert result.name == "env_MY_MAX_LEN"

    def test_differing_values_join_in_order(self):
        """Order matters -- index N of the list is pod index N's value.

        Reversing it is not a cosmetic error: the pods still come up healthy,
        still carry their context-length labels, and each one serves the other
        one's window.
        """
        env = {"MY_MAX_SEQS": "64,,16"}
        assert BaseSmoketest.assert_env_variant_list(
            env, "MY_MAX_SEQS", [64, 16]
        ).passed
        # Reversed must NOT pass: decode-0 and decode-1 would be swapped.
        assert not BaseSmoketest.assert_env_variant_list(
            env, "MY_MAX_SEQS", [16, 64]
        ).passed

    def test_wrong_value_fails_with_both_sides(self):
        env = {"MY_MAX_LEN": "2048,,2048"}
        result = BaseSmoketest.assert_env_variant_list(env, "MY_MAX_LEN", [2048, 4096])
        assert not result.passed
        assert result.expected == "2048,,4096"
        assert result.actual == "2048,,2048"

    def test_missing_env_var_fails(self):
        result = BaseSmoketest.assert_env_variant_list({}, "MY_MAX_LEN", [2048, 2048])
        assert not result.passed
        assert result.actual == "not set"

    def test_single_variant_has_no_delimiter(self):
        """One variant renders as a bare scalar, not ``2048,,``."""
        env = {"MY_MAX_LEN": "2048"}
        assert BaseSmoketest.assert_env_variant_list(env, "MY_MAX_LEN", [2048]).passed

    def test_omitted_key_renders_as_empty_segment(self):
        """A replica that should keep the engine's own default gets no value.

        The scenario writes the gap as an empty segment, so ``None`` in the
        expected list has to compare equal to "" -- not to the literal "None".
        """
        env = {"MY_MAX_SEQS": "64,,"}
        assert BaseSmoketest.assert_env_variant_list(
            env, "MY_MAX_SEQS", [64, None]
        ).passed

    def test_message_mentions_variant_count(self):
        """The failure has to explain why a ,, list is expected.

        Without it the next reader sees ``2048,,4096`` in a pod spec and
        diagnoses a rendering bug, which is exactly the wrong turn this helper
        was written to prevent.
        """
        env = {"MY_MAX_LEN": "2048,,2048"}
        result = BaseSmoketest.assert_env_variant_list(env, "MY_MAX_LEN", [2048, 4096])
        assert "2 variants" in result.message
        assert "pod index" in result.message


# ---------------------------------------------------------------------------
# validate_role_pods
# ---------------------------------------------------------------------------

#: The command the scenario states. Written the way a user would: engine
#: parameters inline, the per-replica one behind an env var they named.
DECODE_COMMAND = (
    "vllm serve $MODEL_NAME \\\n"
    "  --port 8200 \\\n"
    "  --max-model-len $MY_MAX_LEN \\\n"
    "  --block-size 16\n"
)


def _collect(
    monkeypatch, config: dict, env: dict, args: str | None = None, role: str = "decode"
):
    """Run validate_role_pods against a synthetic pod.

    Returns ``{check_name: CheckResult}``. ``args`` defaults to the container
    actually carrying ``DECODE_COMMAND``, i.e. the healthy case.
    """
    from llmdbenchmark.smoketests.base import SmoketestReport

    smoketest = BaseSmoketest.__new__(BaseSmoketest)
    pod = {
        "metadata": {"name": f"test-{role}-0", "namespace": "test-ns"},
        "spec": {
            "nodeName": "node-0",
            "containers": [
                {
                    "name": "modelserver",
                    "command": ["/bin/bash", "-c"],
                    "args": [DECODE_COMMAND if args is None else args],
                    "env": [{"name": k, "value": v} for k, v in env.items()],
                    "resources": {},
                }
            ],
        },
    }
    monkeypatch.setattr(
        BaseSmoketest, "get_pod_specs", lambda self, *a, **kw: [pod], raising=True
    )
    report = SmoketestReport()
    smoketest.validate_role_pods(
        cmd=None,
        namespace="test-ns",
        config=config,
        role=role,
        model_short="test-model",
        report=report,
    )
    return {c.name: c for c in report.checks}


def _base_config(role_extra: dict) -> dict:
    role: dict = {"engine": {"command": DECODE_COMMAND, "port": 8200}}
    role.update(role_extra)
    return {
        "model": {"name": "test-model"},
        "engine": {"servicePort": 8000, "containerName": "modelserver"},
        "decode": role,
    }


class TestPerReplicaEnvVarInRolePods:
    """A ``,,`` list in extraEnvVars is validated as the list it is."""

    def test_per_replica_list_passes(self, monkeypatch):
        """The old regression, in its new form.

        Under ``vllmVariants`` the validator compared the rendered
        ``2048,,32768`` against the single scalar ``model.maxModelLen`` and
        failed every context-length-aware scenario. The value is now a plain
        env var the scenario declares, so it is compared against what the
        scenario declared -- there is no second source to disagree with.
        """
        config = _base_config(
            {
                "replicas": 2,
                "contextLengthRanges": ["0-1000", "1000-2048"],
                "extraEnvVars": [{"name": "MY_MAX_LEN", "value": "2048,,32768"}],
            }
        )
        checks = _collect(monkeypatch, config, {"MY_MAX_LEN": "2048,,32768"})
        assert checks["env_MY_MAX_LEN"].passed

    def test_wrong_per_replica_list_still_fails(self, monkeypatch):
        """Not a blanket pass: a dropped segment has to be caught.

        Losing the second segment is the realistic rendering bug -- the second
        replica would then serve the first replica's context window while still
        advertising the wider label range.
        """
        config = _base_config(
            {
                "replicas": 2,
                "extraEnvVars": [{"name": "MY_MAX_LEN", "value": "2048,,32768"}],
            }
        )
        checks = _collect(monkeypatch, config, {"MY_MAX_LEN": "2048"})
        assert not checks["env_MY_MAX_LEN"].passed


class TestCommandDeliveredVerbatim:
    """The command is the contract, so delivering it unchanged is the check."""

    def test_matching_command_passes(self, monkeypatch):
        config = _base_config({"replicas": 1})
        checks = _collect(monkeypatch, config, {"MY_MAX_LEN": "2048"})
        assert checks["engine_command"].passed

    def test_damaged_command_fails(self, monkeypatch):
        """A flag lost between scenario and pod must fail, whatever ate it.

        This is the single check that replaced ~130 lines of per-flag
        assertions, and it has to stay sharp: a shell layer eating a line
        continuation, ``${...}`` left unsubstituted, or a role silently
        inheriting the plan-wide command all show up here.
        """
        config = _base_config({"replicas": 1})
        damaged = DECODE_COMMAND.replace("--max-model-len $MY_MAX_LEN \\\n", "")
        checks = _collect(monkeypatch, config, {}, args=damaged)
        assert not checks["engine_command"].passed
        assert "--max-model-len" in checks["engine_command"].expected

    def test_no_env_check_is_derived_from_an_engine_flag(self, monkeypatch):
        """Regression guard for the verbatim-command contract itself.

        ``--max-model-len`` and ``--block-size`` are in the command above, and
        the pod env deliberately does NOT carry a ``VLLM_MAX_MODEL_LEN`` or
        ``VLLM_BLOCK_SIZE``. If a check named after one appears, someone has
        taught the validator an engine's flags again -- which is the
        maintenance burden this design exists to remove, and it would break the
        moment the role ran SGLang or TRT-LLM instead.
        """
        config = _base_config({"replicas": 1})
        checks = _collect(monkeypatch, config, {"MY_MAX_LEN": "2048"})
        derived = [name for name in checks if name.startswith("env_VLLM_")]
        assert derived == [], f"engine flags leaked back into checks: {derived}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
