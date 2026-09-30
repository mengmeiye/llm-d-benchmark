"""``standalone`` takes decode's pod shape when it states none of its own.

``standalone.engine.command`` already inherits ``decode.engine.command``:
standalone is decode without the router in front of it, so it serves the same
model with the same engine and the same flags. What did not inherit was the pod
that command needs to run, and the two came from different places -- the command
from the scenario, the pod from ``defaults.yaml``. Three ways that bit:

* ``guides/optimized-baseline`` sizes decode at 64Gi and standalone fell back to
  the 40Gi floor. On a real cluster the TRT-LLM pod was OOMKilled 73s in,
  loading a 32B checkpoint whose weights the capacity planner had already
  measured at 61.02 GB;
* ``guides/turn-priority-fairness`` gives decode four devices, and the command it
  hands standalone asks the engine for four -- against a one-device pod;
* the same run's ``LD_LIBRARY_PATH``, which the llm-d TRT-LLM guide requires and
  the ``# @engine trtllm`` switch duly wrote into ``decode.extraEnvVars``, never
  reached the standalone container at all.

``examples/spyre-s390x.yaml`` restates all three by hand, which is what the
missing rule cost before it existed.

The rules pinned here are the ones no rendered manifest would show to be wrong:
it inherits only where standalone is silent, it inherits whole rather than
merging, it runs on the scenario layer so absence still means something, and it
reads decode wherever the scenario wrote it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from llmdbenchmark.parser.render_plans import RenderPlans


BIG = {"limits": {"memory": "64Gi", "cpu": "16"}, "requests": {"memory": "64Gi"}}


def _renderer() -> RenderPlans:
    renderer = RenderPlans.__new__(RenderPlans)
    renderer.logger = MagicMock()
    return renderer


def _inherit(layer: dict) -> dict:
    return _renderer()._inherit_standalone_resources(layer)


class TestInheritance:
    def test_resources_reach_a_standalone_that_states_none(self):
        out = _inherit({"decode": {"resources": BIG}, "standalone": {"enabled": True}})
        assert out["standalone"]["resources"] == BIG

    def test_parallelism_reaches_standalone(self):
        # The command inherited from decode asks the engine for four devices;
        # the kubelet has to grant four.
        out = _inherit({"decode": {"parallelism": {"tensor": 4}}, "standalone": {}})
        assert out["standalone"]["parallelism"] == {"tensor": 4}

    def test_extra_env_vars_reach_standalone(self):
        env = [{"name": "LD_LIBRARY_PATH", "value": "/usr/local/tensorrt/lib"}]
        out = _inherit({"decode": {"extraEnvVars": env}, "standalone": {}})
        assert out["standalone"]["extraEnvVars"] == env

    def test_a_layer_with_no_standalone_block_still_gets_one(self):
        # guides/optimized-baseline never mentions standalone, and `-t standalone`
        # is exactly the run that needs decode's sizing.
        out = _inherit({"modelservice": {"decode": {"resources": BIG}}})
        assert out["standalone"]["resources"] == BIG

    def test_decode_is_read_under_modelservice(self):
        out = _inherit({"modelservice": {"decode": {"parallelism": {"tensor": 2}}}})
        assert out["standalone"]["parallelism"] == {"tensor": 2}

    def test_the_nested_spelling_wins_over_the_flat_one(self):
        out = _inherit(
            {
                "decode": {"resources": {"limits": {"memory": "8Gi"}}},
                "modelservice": {"decode": {"resources": BIG}},
            }
        )
        assert out["standalone"]["resources"] == BIG


class TestStandaloneWins:
    def test_a_stated_key_is_left_alone(self):
        own = {"limits": {"memory": "200Gi", "cpu": "4"}}
        out = _inherit({"decode": {"resources": BIG}, "standalone": {"resources": own}})
        assert out["standalone"]["resources"] == own

    def test_inheritance_is_whole_not_merged(self):
        # Naming the key at all opts it out completely: no decode CPU leaks into
        # a standalone block that named only memory. Same convention roleDefaults
        # uses, and the reason defaults.yaml tells authors to state both.
        own = {"limits": {"memory": "200Gi"}}
        out = _inherit({"decode": {"resources": BIG}, "standalone": {"resources": own}})
        assert "cpu" not in out["standalone"]["resources"]["limits"]
        assert "requests" not in out["standalone"]["resources"]

    def test_keys_are_decided_one_by_one(self):
        out = _inherit(
            {
                "decode": {"resources": BIG, "parallelism": {"tensor": 4}},
                "standalone": {"parallelism": {"tensor": 1}},
            }
        )
        assert out["standalone"]["parallelism"] == {"tensor": 1}
        assert out["standalone"]["resources"] == BIG


class TestNoOp:
    def test_nothing_to_inherit_from(self):
        assert _inherit({"standalone": {"enabled": True}}) == {
            "standalone": {"enabled": True}
        }

    def test_no_decode_block_at_all(self):
        assert _inherit({"harness": {"name": "inference-perf"}}) == {
            "harness": {"name": "inference-perf"}
        }

    def test_a_decode_key_written_empty_is_not_inherited(self):
        # A YAML key with no value is `None`, and deep_merge already refuses to
        # let one clobber a default. Inheriting it would smuggle it past that.
        out = _inherit({"decode": {"resources": None}, "standalone": {}})
        assert "resources" not in out["standalone"]

    def test_only_the_named_keys_travel(self):
        out = _inherit(
            {
                "decode": {"replicas": 3, "probes": {"startup": {}}, "resources": BIG},
                "standalone": {},
            }
        )
        assert set(out["standalone"]) == {"resources"}

    def test_the_input_layer_is_not_mutated(self):
        layer = {"decode": {"resources": BIG}}
        _inherit(layer)
        assert "standalone" not in layer

    def test_a_wrongly_shaped_standalone_is_left_for_the_schema(self):
        out = _inherit({"decode": {"resources": BIG}, "standalone": "yes"})
        assert out["standalone"] == "yes"
