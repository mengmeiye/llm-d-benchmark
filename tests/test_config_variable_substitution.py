"""Tests for ${dotted.path} config variable substitution in render_plans.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llmdbenchmark.parser.render_plans import RenderPlans


@pytest.fixture
def renderer():
    """Create a RenderPlans instance with a mock logger."""
    logger = MagicMock()
    logger.log_warning = MagicMock()
    logger.log_info = MagicMock()
    r = RenderPlans.__new__(RenderPlans)
    r.logger = logger
    return r


class TestSubstituteConfigVariables:
    """Tests for _substitute_config_variables()."""

    def test_basic_substitution(self, renderer):
        values = {
            "model": {"name": "meta-llama/Llama-3.1-8B"},
            "field": "served model is ${model.name}",
        }
        result = renderer._substitute_config_variables(values)
        assert result["field"] == "served model is meta-llama/Llama-3.1-8B"

    def test_nested_path(self, renderer):
        values = {
            "model": {"name": "my-model", "path": "models/my-model"},
            "decode": {
                "engine": {
                    "command": "vllm serve /cache/${model.path} --served-model-name ${model.name}"
                }
            },
        }
        result = renderer._substitute_config_variables(values)
        assert (
            result["decode"]["engine"]["command"]
            == "vllm serve /cache/models/my-model --served-model-name my-model"
        )

    def test_shell_vars_left_alone(self, renderer):
        """Shell variables in an engine command must reach the container intact.

        The command is passed through verbatim, so the only thing this pass may
        touch is a dotted ``${a.b}`` reference. ``$MODEL_SERVE_REF`` (exported
        into every serving pod) and ``$LWS_WORKER_INDEX`` (set by LeaderWorkerSet)
        resolve at pod start, not at render time.
        """
        values = {
            "model": {"name": "test-model"},
            "field": (
                "vllm serve $MODEL_SERVE_REF --served-model-name ${model.name} "
                "--data-parallel-rank $LWS_WORKER_INDEX"
            ),
        }
        result = renderer._substitute_config_variables(values)
        assert result["field"] == (
            "vllm serve $MODEL_SERVE_REF --served-model-name test-model "
            "--data-parallel-rank $LWS_WORKER_INDEX"
        )

    def test_shell_braced_vars_left_alone(self, renderer):
        """${SINGLE_WORD} without dots should NOT be substituted.

        Includes the ``:-default`` form, which wide-ep commands rely on.
        """
        values = {
            "field": "rank is ${LWS_WORKER_INDEX:-0} of ${LWS_GROUP_SIZE}",
        }
        result = renderer._substitute_config_variables(values)
        assert result["field"] == "rank is ${LWS_WORKER_INDEX:-0} of ${LWS_GROUP_SIZE}"

    def test_unresolvable_ref_left_as_is(self, renderer):
        values = {"field": "value is ${nonexistent.key}"}
        result = renderer._substitute_config_variables(values)
        assert result["field"] == "value is ${nonexistent.key}"
        renderer.logger.log_warning.assert_called_once()

    def test_non_string_values_unchanged(self, renderer):
        values = {
            "model": {"name": "test"},
            "count": 42,
            "enabled": True,
            "ratio": 0.5,
        }
        result = renderer._substitute_config_variables(values)
        assert result["count"] == 42
        assert result["enabled"] is True
        assert result["ratio"] == 0.5

    def test_list_of_env_vars(self, renderer):
        values = {
            "model": {"name": "my-org/my-model"},
            "extraEnvVars": [
                {"name": "MODEL", "value": "${model.name}"},
                {"name": "PORT", "value": "8000"},
            ],
        }
        result = renderer._substitute_config_variables(values)
        assert result["extraEnvVars"][0]["value"] == "my-org/my-model"
        assert result["extraEnvVars"][1]["value"] == "8000"

    def test_multiple_refs_in_one_string(self, renderer):
        values = {
            "model": {"name": "my-model", "path": "models/my-model"},
            "field": "${model.name} at ${model.path}",
        }
        result = renderer._substitute_config_variables(values)
        assert result["field"] == "my-model at models/my-model"

    def test_dict_or_list_value_not_substituted(self, renderer):
        """Dotted paths resolving to dicts or lists should not be substituted."""
        values = {
            "model": {"name": "test", "nested": {"deep": "value"}},
            "field": "ref is ${model.nested}",
        }
        result = renderer._substitute_config_variables(values)
        assert result["field"] == "ref is ${model.nested}"

    def test_original_values_not_mutated(self, renderer):
        values = {
            "model": {"name": "original"},
            "field": "${model.name}",
        }
        result = renderer._substitute_config_variables(values)
        assert result["field"] == "original"
        assert values["field"] == "${model.name}"

    def test_numeric_value_substituted_as_string(self, renderer):
        values = {
            "model": {"maxModelLen": 32768},
            "field": "max len is ${model.maxModelLen}",
        }
        result = renderer._substitute_config_variables(values)
        assert result["field"] == "max len is 32768"


class TestModelIdLabelAlias:
    """Tests that _resolve_model_id_label exposes model_id_label as model.idLabel."""

    def test_model_id_label_available_as_dotted_path(self, renderer):
        """${model.idLabel} resolves to the computed model_id_label value."""
        values = {
            "model": {"name": "meta-llama/Llama-3.2-1B-Instruct"},
            "namespace": {"name": "benchmark"},
            "extraObjects": [
                {
                    "apiVersion": "inference.networking.x-k8s.io/v1alpha2",
                    "kind": "InferenceObjective",
                    "spec": {"poolRef": {"name": "${model.idLabel}-router"}},
                }
            ],
        }
        resolved = renderer._resolve_model_id_label(values)
        result = renderer._substitute_config_variables(resolved)

        expected_label = resolved["model_id_label"]
        assert expected_label, "model_id_label should be computed"
        assert (
            result["extraObjects"][0]["spec"]["poolRef"]["name"]
            == f"{expected_label}-router"
        )
