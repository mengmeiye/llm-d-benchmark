"""Appending words to a shared engine command.

A scenario states its launch line verbatim, which is the whole of this project's
engine-agnosticism: no flag is named, typed or validated here, so a new engine is
not a change to llm-d-benchmark. That works because one stack has one command.

Multi-model scenarios break the "one" -- several stacks share ONE command and a
few of them differ in a flag or two -- and the alternatives are both bad: restate
sixteen identical lines per stack, or start modelling the flags that differ, which
is the design this project replaced. ``engine.extraArgs`` is the third option:
words concatenated onto the end of the command, unexamined.

These tests pin the two properties that make it safe. Nothing is parsed, so the
maintenance surface does not grow; and a flag stated twice is read back as the
occurrence the engine itself will use, so the numbers the capacity check and the
router's prefix-cache index run on are the numbers the process gets.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from llmdbenchmark.engine import compose_command, parse_command, resolve_engines
from llmdbenchmark.parser.cluster_resource_resolver import ClusterResourceResolver
from llmdbenchmark.parser.render_plans import RenderPlans
from llmdbenchmark.parser.version_resolver import VersionResolver


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = PROJECT_ROOT / "config" / "templates" / "jinja"
DEFAULTS = PROJECT_ROOT / "config" / "templates" / "values" / "defaults.yaml"
MULTI_MODEL = (
    PROJECT_ROOT
    / "config"
    / "scenarios"
    / "examples"
    / "multi-model-optimized-baseline.yaml"
)

COMMAND = "vllm serve m \\\n  --port 8200 \\\n  --max-model-len 8192\n"


def _resolve(values):
    return resolve_engines(values), values


# ---------------------------------------------------------------------------
# composing
# ---------------------------------------------------------------------------


def test_words_are_appended_to_the_command_as_a_continuation():
    """Still one shell command, and still readable in the rendered manifest."""
    composed = compose_command(COMMAND, ["--max-model-len", "4096"])

    assert composed == (
        "vllm serve m \\\n  --port 8200 \\\n  --max-model-len 8192 \\\n"
        "  --max-model-len 4096"
    )
    # And it is one command, not two: the appended words are read as part of the
    # same launch segment, not as a second command after a separator.
    parsed = parse_command(composed)
    assert parsed.preamble == ""
    assert parsed.model == "m"
    assert parsed.flags["--max-model-len"] == "4096"


def test_nothing_to_append_leaves_the_command_identical():
    """The overwhelmingly common case must not even reformat the text: a
    scenario with no extraArgs renders exactly what it wrote, which is what the
    verbatim contract and the smoketest that checks it both mean."""
    assert compose_command(COMMAND, []) == COMMAND
    assert compose_command(COMMAND, [None, ""]) == COMMAND


def test_a_single_line_command_gets_a_two_space_continuation():
    assert compose_command("vllm serve m --port 8200", ["--dtype", "bfloat16"]) == (
        "vllm serve m --port 8200 \\\n  --dtype bfloat16"
    )


def test_a_trailing_continuation_is_not_doubled():
    """A command that already ends in a backslash -- easy to leave behind when
    editing a block scalar -- must not produce `\\ \\`, which is a literal
    backslash argument followed by a line join."""
    composed = compose_command(
        "vllm serve m \\\n  --port 8200 \\\n", ["--dtype", "bf16"]
    )

    assert composed.count("\\\n") == 2
    assert parse_command(composed).flags["--dtype"] == "bf16"


def test_no_command_means_nothing_to_extend():
    assert compose_command(None, ["--dtype", "bfloat16"]) is None


@pytest.mark.parametrize(
    "token,expected",
    [
        ("--max-model-len", "--max-model-len"),
        ("bfloat16", "bfloat16"),
        ("/health,/metrics", "/health,/metrics"),
        ("--disable-log-for=/a,/b", "--disable-log-for=/a,/b"),
        # Would otherwise split into two arguments.
        ("a b", "'a b'"),
        # Already spelled as shell by the user: a variable reference must stay
        # expandable, and their own quoting must not be quoted again.
        ("$ENGINE_PORT", "$ENGINE_PORT"),
        ("'{\"x\": 1}'", "'{\"x\": 1}'"),
    ],
)
def test_a_word_reaches_the_shell_as_the_user_meant_it(token, expected):
    assert compose_command("vllm serve m", [token]).endswith(expected)


# ---------------------------------------------------------------------------
# what the rest of the pipeline then sees
# ---------------------------------------------------------------------------


def test_the_composed_line_is_what_templates_render():
    """The command written back to the values tree is the composed one, so every
    consumer -- the modelservice values, the standalone Deployment, the nok8s
    container, the smoketest that asserts the command reached the pod intact --
    sees one string and none of them needs to know about extraArgs."""
    _, values = _resolve(
        {
            "model": {},
            "decode": {
                "engine": {
                    "command": COMMAND,
                    "extraArgs": ["--tensor-parallel-size", "2"],
                }
            },
        }
    )

    assert values["decode"]["engine"]["command"].endswith("--tensor-parallel-size 2")


def test_a_repeated_flag_reads_back_as_the_one_the_engine_will_use():
    """This is the property the whole design rests on.

    ``--max-model-len`` appears twice, and the engine's own argument parser keeps
    the last -- argparse and click both do. If the read-back kept the first, the
    pre-deploy capacity check would size KV cache for 8192 while the process ran
    at 4096, and the router's prefix-cache index would hash on a page size the
    engine never writes. So the reader has to agree with the engine, not with the
    order the text happens to be in.
    """
    _, values = _resolve(
        {
            "model": {},
            "decode": {
                "engine": {
                    "command": COMMAND + "  --block-size 64\n",
                    "extraArgs": ["--max-model-len", "4096", "--block-size", "16"],
                }
            },
        }
    )

    assert values["model"]["maxModelLen"] == 4096
    assert values["model"]["blockSize"] == 16


def test_an_appended_port_moves_the_service_and_the_probes_with_it():
    """Reading the port off the command is what keeps the container port, the
    probes and the routing sidecar's upstream in step; an appended --port has to
    move them too, or the pod comes up healthy on a port nothing talks to."""
    _, values = _resolve(
        {
            "model": {},
            "decode": {"engine": {"command": COMMAND, "extraArgs": ["--port", "9000"]}},
        }
    )

    assert values["decode"]["engine"]["port"] == 9000


def test_the_model_id_is_still_read_from_the_command():
    """extraArgs is appended at the end, so it cannot reach the serve target --
    the positional that ``model.name`` is derived from. That limit is the reason
    ``${model.name}`` stays in a shared multi-model command."""
    _, values = _resolve(
        {
            "model": {},
            "decode": {
                "engine": {
                    "command": "vllm serve facebook/opt-125m --port 8200",
                    "extraArgs": ["--dtype", "bfloat16"],
                }
            },
        }
    )

    assert values["resolvedServingRoles"]["decode"]["modelId"] == "facebook/opt-125m"


# ---------------------------------------------------------------------------
# where the words come from
# ---------------------------------------------------------------------------


def test_a_roles_own_list_replaces_the_plan_wide_one():
    """Same rule as ``command``: one place states the words for a role, so there
    is never a question of which order two lists concatenate in."""
    _, values = _resolve(
        {
            "model": {},
            "engine": {"command": COMMAND, "extraArgs": ["--dtype", "float16"]},
            "decode": {"engine": {"extraArgs": ["--dtype", "bfloat16"]}},
            "prefill": {"engine": {}},
        }
    )

    assert values["decode"]["engine"]["command"].endswith("--dtype bfloat16")
    assert values["prefill"]["engine"]["command"].endswith("--dtype float16")


def test_standalone_does_not_repeat_common_args_from_inherited_decode():
    _, values = _resolve(
        {
            "model": {},
            "engine": {"extraArgs": ["--api-key", "secret"]},
            "decode": {"engine": {"command": COMMAND}},
            "standalone": {"enabled": True, "engine": {}},
        }
    )

    command = values["standalone"]["engine"]["command"]
    assert command.count("--api-key secret") == 1


def test_standalone_specific_args_extend_the_inherited_composed_command():
    _, values = _resolve(
        {
            "model": {},
            "engine": {"extraArgs": ["--dtype", "float16"]},
            "decode": {"engine": {"command": COMMAND}},
            "standalone": {
                "enabled": True,
                "engine": {"extraArgs": ["--dtype", "bfloat16"]},
            },
        }
    )

    command = values["standalone"]["engine"]["command"]
    assert "--dtype float16" in command
    assert command.endswith("--dtype bfloat16")


def test_words_with_no_command_to_extend_are_reported():
    """A role whose image entrypoint launches the server has no command here, so
    appended words would silently vanish. ``engine.args`` is the field that
    reaches an entrypoint, and the warning says so."""
    warnings, values = _resolve(
        {
            "model": {},
            "decode": {"engine": {"command": None, "extraArgs": ["--dtype", "bf16"]}},
        }
    )

    assert values["decode"]["engine"]["command"] is None
    assert any("extraArgs is set but there is no command" in w for w in warnings)
    assert any("decode.engine.args" in w for w in warnings)


def test_something_that_is_not_a_list_is_reported_not_dropped():
    """``extraArgs: "--max-model-len 4096"`` is a natural mistake. Splitting it
    would be guessing at the user's quoting, so it is refused -- but loudly,
    because the alternative is a stack that runs without the flags its author
    thought they had set."""
    warnings, _ = _resolve(
        {
            "model": {},
            "decode": {
                "engine": {"command": COMMAND, "extraArgs": "--max-model-len 4096"}
            },
        }
    )

    assert any("must be a list of words" in w for w in warnings)


# ---------------------------------------------------------------------------
# the scenario shape this exists for
# ---------------------------------------------------------------------------


def _render(tmp_path, scenario=MULTI_MODEL):
    logger = MagicMock()
    renderer = RenderPlans(
        template_dir=TEMPLATES,
        defaults_file=DEFAULTS,
        scenarios_file=scenario,
        output_dir=tmp_path / "plans",
        logger=logger,
        setup_overrides={},
        version_resolver=VersionResolver(logger=logger, dry_run=True),
        cluster_resource_resolver=ClusterResourceResolver(logger=logger, dry_run=True),
    )
    result = renderer.eval()
    return {
        yaml.safe_load((path / "config.yaml").read_text())["model"]["name"]: path
        for path in result.rendered_paths
    }


def test_the_shared_command_states_every_flag_once(tmp_path):
    """Two stacks, one command, and the numbers outside the engine read back off
    it -- so nothing in this scenario states a context length or a page size
    twice, and the `model:` blocks carry only what identifies a model."""
    plans = _render(tmp_path)
    assert len(plans) == 2

    for path in plans.values():
        merged = yaml.safe_load((path / "config.yaml").read_text())
        command = merged["decode"]["engine"]["command"]

        assert merged["model"]["name"] in command
        assert merged["model"]["maxModelLen"] == 8192
        assert merged["model"]["blockSize"] == 64
        assert merged["model"]["gpuMemoryUtilization"] == 0.95


def test_one_stack_can_differ_by_two_words(tmp_path):
    """The point of the whole feature, end to end through the renderer.

    Written the way a user writes it -- two words under one stack's
    ``decode.engine`` -- rather than through ``--set``, because the authoring
    experience is the thing being tested. It reaches that stack's rendered
    command and its read-back numbers, and leaves the other stack alone.
    """
    scenario = yaml.safe_load(MULTI_MODEL.read_text())
    stack = scenario["scenario"][1]
    assert stack["name"] == "llama-31-8b"
    stack["modelservice"]["decode"]["engine"] = {
        "extraArgs": ["--max-model-len", "4096"]
    }
    edited = tmp_path / "edited-scenario.yaml"
    edited.write_text(yaml.safe_dump(scenario, sort_keys=False))

    plans = _render(tmp_path, scenario=edited)

    changed = yaml.safe_load(
        (plans["unsloth/Meta-Llama-3.1-8B"] / "config.yaml").read_text()
    )
    untouched = yaml.safe_load((plans["Qwen/Qwen3-0.6B"] / "config.yaml").read_text())

    assert changed["decode"]["engine"]["command"].endswith("--max-model-len 4096")
    assert changed["model"]["maxModelLen"] == 4096
    assert untouched["model"]["maxModelLen"] == 8192
    assert "4096" not in untouched["decode"]["engine"]["command"]
