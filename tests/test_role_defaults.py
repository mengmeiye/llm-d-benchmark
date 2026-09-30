"""``roleDefaults:`` states a role's shape once instead of once per role.

decode and prefill describe the same pod. A scenario configuring both wrote
every field twice -- the same ``initContainers``, the same RDMA
``securityContext``, the same NIXL environment -- and where one copy was
updated and the other was not, the two roles drifted silently.
``roleDefaults:`` holds what the roles share; each role carries only its
differences.

The rules these tests pin down, because each one is a place the expansion could
be wrong in a way no rendered manifest would reveal:

* it seeds, it does not override -- a key named under ``decode:`` wins;
* it seeds only where the target role models the key, so ``shm`` reaches decode
  and not prefill;
* a key no role models is a typo and raises, rather than being dropped;
* values replace, lists included, so shared plumbing lives in one place only;
* a role the layer never mentions is not conjured into existence.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from llmdbenchmark.parser.render_plans import RenderPlans


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_FILE = ROOT / "config" / "templates" / "values" / "defaults.yaml"


def _renderer() -> RenderPlans:
    renderer = RenderPlans.__new__(RenderPlans)
    renderer.logger = MagicMock()
    return renderer


@pytest.fixture(scope="module")
def defaults() -> dict:
    values = yaml.safe_load(DEFAULTS_FILE.read_text(encoding="utf-8"))
    # Guard the guard: the key filter is driven by these sections, so an empty
    # read would make every "seeded" assertion below pass for the wrong reason.
    assert set(values["decode"]) >= {"resources", "extraEnvVars", "shm"}
    return values


class TestNoOpWhenAbsent:
    def test_a_layer_without_role_defaults_is_returned_unchanged(self) -> None:
        values = {"decode": {"replicas": 2}, "prefill": {"replicas": 1}}

        assert _renderer()._expand_role_defaults(deepcopy(values)) == values

    def test_the_input_is_never_mutated(self) -> None:
        values = {
            "roleDefaults": {"extraEnvVars": [{"name": "A", "value": "1"}]},
            "decode": {"replicas": 2},
        }
        before = deepcopy(values)

        _renderer()._expand_role_defaults(values)

        assert values == before

    def test_the_key_does_not_survive_expansion(self) -> None:
        """It is an authoring convenience, so nothing downstream should see it."""
        result = _renderer()._expand_role_defaults(
            {"roleDefaults": {"replicas": 1}, "decode": {}}
        )

        assert "roleDefaults" not in result


class TestSeeding:
    def test_both_roles_receive_the_shared_block(self) -> None:
        result = _renderer()._expand_role_defaults(
            {
                "roleDefaults": {"initContainers": [{"name": "preprocess"}]},
                "decode": {"replicas": 2},
                "prefill": {"replicas": 1},
            }
        )

        for role in ("decode", "prefill"):
            assert result[role]["initContainers"] == [{"name": "preprocess"}]

    def test_a_role_that_names_the_key_wins(self) -> None:
        result = _renderer()._expand_role_defaults(
            {
                "roleDefaults": {"resources": {"limits": {"memory": "16Gi"}}},
                "decode": {"resources": {"limits": {"memory": "128Gi"}}},
                "prefill": {},
            }
        )

        assert result["decode"]["resources"]["limits"]["memory"] == "128Gi"
        assert result["prefill"]["resources"]["limits"]["memory"] == "16Gi"

    def test_the_merge_is_per_leaf_not_per_block(self) -> None:
        """Overriding limits.memory must not discard the shared requests."""
        result = _renderer()._expand_role_defaults(
            {
                "roleDefaults": {
                    "resources": {
                        "limits": {"memory": "16Gi", "cpu": "8"},
                        "requests": {"memory": "16Gi", "cpu": "8"},
                    }
                },
                "decode": {"resources": {"limits": {"memory": "128Gi"}}},
            }
        )

        assert result["decode"]["resources"] == {
            "limits": {"memory": "128Gi", "cpu": "8"},
            "requests": {"memory": "16Gi", "cpu": "8"},
        }

    def test_a_role_written_empty_is_opting_in(self) -> None:
        """`decode:` with nothing under it means "the shared shape, as is"."""
        for empty in ({}, None):
            result = _renderer()._expand_role_defaults(
                {"roleDefaults": {"replicas": 3}, "decode": empty}
            )
            assert result["decode"]["replicas"] == 3

    def test_a_role_the_layer_never_mentions_is_not_created(self) -> None:
        result = _renderer()._expand_role_defaults(
            {"roleDefaults": {"replicas": 3}, "decode": {}}
        )

        assert "prefill" not in result
        assert "standalone" not in result

    def test_role_blocks_nested_under_modelservice_are_seeded(self) -> None:
        """Role blocks are written there and hoisting runs later."""
        result = _renderer()._expand_role_defaults(
            {
                "roleDefaults": {"replicas": 3},
                "modelservice": {"enabled": True, "decode": {}, "prefill": {}},
            }
        )

        assert result["modelservice"]["decode"]["replicas"] == 3
        assert result["modelservice"]["prefill"]["replicas"] == 3

    def test_the_block_may_itself_be_nested_under_modelservice(self) -> None:
        result = _renderer()._expand_role_defaults(
            {"modelservice": {"roleDefaults": {"replicas": 3}, "decode": {}}}
        )

        assert result["modelservice"]["decode"]["replicas"] == 3
        assert "roleDefaults" not in result["modelservice"]


class TestKeyFiltering:
    def test_a_key_reaches_only_the_roles_that_model_it(self, defaults) -> None:
        """``shm`` is a decode field; prefill has no such key to fill."""
        assert "shm" in defaults["decode"]
        assert "shm" not in defaults["prefill"]

        result = _renderer()._expand_role_defaults(
            {
                "roleDefaults": {"shm": {"size": "16Gi"}, "replicas": 1},
                "decode": {},
                "prefill": {},
            },
            defaults,
        )

        assert result["decode"]["shm"] == {"size": "16Gi"}
        assert "shm" not in result["prefill"]
        assert result["prefill"]["replicas"] == 1

    def test_a_key_no_role_models_is_a_typo_and_raises(self, defaults) -> None:
        with pytest.raises(ValueError, match="resoures"):
            _renderer()._expand_role_defaults(
                {"roleDefaults": {"resoures": {}}, "decode": {}}, defaults
            )

    def test_the_error_lists_what_the_author_could_have_written(self, defaults) -> None:
        with pytest.raises(ValueError, match="resources") as excinfo:
            _renderer()._expand_role_defaults(
                {"roleDefaults": {"resoures": {}}, "decode": {}}, defaults
            )

        assert "extraEnvVars" in str(excinfo.value)

    def test_without_defaults_nothing_is_filtered_or_rejected(self) -> None:
        """The filter is a property of defaults.yaml, not of the expansion."""
        result = _renderer()._expand_role_defaults(
            {"roleDefaults": {"whatever": 1}, "decode": {}}
        )

        assert result["decode"] == {"whatever": 1}

    def test_a_non_mapping_block_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="'roleDefaults' must be a mapping"):
            _renderer()._expand_role_defaults({"roleDefaults": "invalid"})


class TestListsReplace:
    def test_a_role_list_replaces_the_shared_one(self) -> None:
        """Documented, not accidental: appending is a separate change.

        Shared plumbing a role must keep therefore belongs in ``roleDefaults``
        only. A role naming the same list is saying "not that list, this one".
        """
        result = _renderer()._expand_role_defaults(
            {
                "roleDefaults": {"extraEnvVars": [{"name": "SHARED", "value": "1"}]},
                "decode": {"extraEnvVars": [{"name": "MINE", "value": "2"}]},
                "prefill": {},
            }
        )

        assert result["decode"]["extraEnvVars"] == [{"name": "MINE", "value": "2"}]
        assert result["prefill"]["extraEnvVars"] == [{"name": "SHARED", "value": "1"}]


class TestSectionList:
    def test_nok8s_is_not_a_role(self, defaults) -> None:
        """It shares `enabled` and `engine` and nothing else -- it is not a pod.

        Seeding it from a pod-shaped block would put probes and resources on an
        ssh transport, so it is deliberately excluded.
        """
        assert "nok8s" not in RenderPlans._ROLE_DEFAULT_SECTIONS

        nok8s_keys = set(defaults["nok8s"])
        decode_keys = set(defaults["decode"])
        assert nok8s_keys & decode_keys == {"enabled", "engine"}

    def test_every_listed_section_exists_in_defaults(self, defaults) -> None:
        for section in RenderPlans._ROLE_DEFAULT_SECTIONS:
            assert isinstance(defaults.get(section), dict), (
                f"{section} is seeded by roleDefaults but defaults.yaml has no "
                f"such section -- the key filter would drop everything"
            )
