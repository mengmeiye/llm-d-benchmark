"""The commented-out engine commands scenarios carry, actually applied.

A scenario may offer another engine as a comment for the reader to switch in --
`config/scenarios/guides/optimized-baseline.yaml` offers SGLang and TRT-LLM that
way. A commented block is covered by nothing: it is not parsed, not rendered and
not deployed, so it can drift out of correctness silently while the file that
holds it keeps passing every test.

`# @engine <name>` tags each such group, `llmdbenchmark.engine.alternatives`
performs the switch (uncomment the tagged groups, drop the definitions they
replace) for `--engine <name>`, `util/scenario-inventory.py --apply` and these
tests alike. What they hold the alternatives to:

  * the switched command launches the engine the tag names, and the image
    follows it without the scenario naming one;
  * every capacity number the pre-deploy check needs is present -- the failure
    a forgotten companion block causes is a *silently skipped* check, not an
    error, which is why it is asserted directly;
  * the model, the port and the context length match what the active command
    renders -- a context length means the same thing to every engine, unlike
    the memory fraction and the KV page size, which each engine measures its
    own way (see `_PRESERVED`).

So the four TRT-LLM groups cannot be reduced to three, and a flag renamed
upstream is a failing test rather than a comment nobody ran.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from llmdbenchmark.engine import apply_alternative, is_known_engine
from llmdbenchmark.parser.cluster_resource_resolver import ClusterResourceResolver
from llmdbenchmark.parser.render_plans import RenderPlans
from llmdbenchmark.parser.version_resolver import VersionResolver

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCENARIOS = _REPO_ROOT / "config" / "scenarios"
_TEMPLATES = _REPO_ROOT / "config" / "templates" / "jinja"
_DEFAULTS = _REPO_ROOT / "config" / "templates" / "values" / "defaults.yaml"

#: Numbers the capacity planner needs. A switch that loses one of these does not
#: fail -- the check skips its KV arithmetic -- so each is asserted present.
_CAPACITY = ("maxModelLen", "gpuMemoryUtilization", "blockSize")

#: Numbers an alternative is expected to keep, because they mean the same thing
#: in every engine: a context length is a number of tokens whatever serves it.
#:
#: Two capacity numbers are deliberately absent, because the engines do not
#: agree on what they measure -- an alternative may state its own, it just may
#: not drop it (each is asserted present in `_CAPACITY` above):
#:   `blockSize`  a sensible KV page size differs by engine (vLLM 64, TRT-LLM
#:                32, SGLang's `--page-size`).
#:   `gpuMemoryUtilization`  vLLM's `--gpu-memory-utilization` is "the fraction
#:                of GPU memory to be used for the model executor" -- weights,
#:                activations and KV together. SGLang's `--mem-fraction-static`
#:                is "the fraction of the memory used for static allocation
#:                (model weights and KV cache memory pool)", with activations
#:                and CUDA graphs on top. Equal numbers describe *different*
#:                splits, and asserting them equal is what kept
#:                guides/optimized-baseline's SGLang block at 0.95 -- a value
#:                that OOMs mid-forward on one 80GiB device.
_PRESERVED = ("maxModelLen",)

#: The tag itself, as a whole line -- prose that merely quotes `# @engine x` is
#: documentation, not a group.
_TAG_LINE = re.compile(r"^[ \t]*#[ \t]*@engine[ \t]+(\S+)[ \t]*$", re.MULTILINE)

#: A block-scalar launch command. `command:` also names an initContainer's argv
#: list, which is not what an alternative replaces.
_LAUNCH = re.compile(r"^[ \t]*command: \|[ \t]*$", re.MULTILINE)


def _alternatives() -> list[tuple[str, str]]:
    """``(scenario path relative to config/scenarios, engine)`` for every tag.

    Collected by reading the files with a regex of this file's own, so collection
    stays independent of the module under test and a new alternative is picked up
    with no edit here.
    """
    found = []
    for path in sorted(_SCENARIOS.rglob("*.yaml")):
        engines = sorted(set(_TAG_LINE.findall(path.read_text())))
        rel = path.relative_to(_SCENARIOS).with_suffix("").as_posix()
        found.extend((rel, engine.lower()) for engine in engines)
    return found


ALTERNATIVES = _alternatives()


def _render(tmp_path: Path, scenario_text: str) -> dict:
    """Render one scenario's text and return the first stack's config.yaml."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    scenario_file = tmp_path / "scenario.yaml"
    scenario_file.write_text(scenario_text)
    logger = MagicMock()
    result = RenderPlans(
        template_dir=_TEMPLATES,
        defaults_file=_DEFAULTS,
        scenarios_file=scenario_file,
        output_dir=tmp_path / "out",
        logger=logger,
        setup_overrides={},
        version_resolver=VersionResolver(logger=logger, dry_run=True),
        cluster_resource_resolver=ClusterResourceResolver(logger=logger, dry_run=True),
    ).eval()
    stack_dir = Path(result.rendered_paths[0])
    return yaml.safe_load((stack_dir / "config.yaml").read_text())


def _serving_role(config: dict) -> dict:
    """The role whose engine the alternative switched: the first one rendered."""
    for role in ("decode", "standalone", "prefill", "nok8s"):
        section = config.get(role)
        if isinstance(section, dict) and isinstance(section.get("engine"), dict):
            if section["engine"].get("command"):
                return section
    raise AssertionError("no role with a launch command was rendered")


def test_there_is_at_least_one_alternative_to_check():
    """A regex that stops matching would otherwise make this file vacuous."""
    assert ALTERNATIVES, "no `# @engine` tags found under config/scenarios/"


@pytest.mark.parametrize(("scenario", "engine"), ALTERNATIVES, ids=lambda v: str(v))
def test_tag_names_a_known_engine(scenario, engine):
    """An unknown name would select nothing and silently test nothing."""
    assert is_known_engine(engine), (
        f"{scenario}.yaml tags `# @engine {engine}`, which no EngineSpec claims"
    )


@pytest.mark.parametrize(("scenario", "engine"), ALTERNATIVES, ids=lambda v: str(v))
def test_alternative_applies_and_parses(scenario, engine):
    text = (_SCENARIOS / f"{scenario}.yaml").read_text()
    switched = apply_alternative(text, engine)

    # The tag is consumed by the switch: one left behind would mean a group was
    # found and not applied. Other engines' tags stay -- their groups still do.
    assert engine not in _TAG_LINE.findall(switched)
    assert yaml.safe_load(switched), f"{scenario} -> {engine} rendered no document"


@pytest.mark.parametrize(("scenario", "engine"), ALTERNATIVES, ids=lambda v: str(v))
def test_alternative_renders_the_engine_it_claims(tmp_path, scenario, engine):
    text = (_SCENARIOS / f"{scenario}.yaml").read_text()
    switched = _render(tmp_path / "switched", apply_alternative(text, engine))
    role = _serving_role(switched)

    # Detected from the command, not declared anywhere: if the launcher stopped
    # being recognised, the engine would fall back and the image with it.
    assert role["engine"]["facts"]["engine"] == engine
    # Which `images.<key>` entry the pod gets, chosen by the detected engine --
    # the scenario names no image at all.
    assert role["engine"]["imageKey"] == engine
    assert role["engine"]["image"]["repository"]


@pytest.mark.parametrize(("scenario", "engine"), ALTERNATIVES, ids=lambda v: str(v))
def test_alternative_keeps_the_facts_the_active_command_states(
    tmp_path, scenario, engine
):
    text = (_SCENARIOS / f"{scenario}.yaml").read_text()
    active = _render(tmp_path / "active", text)
    switched = _render(tmp_path / "switched", apply_alternative(text, engine))

    # A companion block left behind does not fail the render: the number simply
    # arrives as None and the capacity check skips itself. Assert it directly.
    missing = [key for key in _CAPACITY if switched["model"].get(key) is None]
    assert not missing, (
        f"{scenario} -> {engine} lost {missing}: the launch command does not "
        f"state them in this engine's spelling and no `# @engine {engine}` group "
        f"supplies them either"
    )

    for key in _PRESERVED:
        assert switched["model"][key] == active["model"][key], (
            f"{scenario} -> {engine} changed model.{key}; an alternative is "
            f"the same workload on another engine, so this value should match"
        )

    active_role, switched_role = _serving_role(active), _serving_role(switched)
    assert switched_role["engine"]["port"] == active_role["engine"]["port"]
    assert (
        switched_role["engine"]["facts"]["model"]
        == active_role["engine"]["facts"]["model"]
    )


def test_applying_an_engine_with_no_group_is_an_error():
    """Silence here would report a switch that never happened as a pass."""
    text = (_SCENARIOS / "guides/optimized-baseline.yaml").read_text()
    with pytest.raises(ValueError, match="no .# @engine nosuchengine. group"):
        apply_alternative(text, "nosuchengine")


def test_the_switch_removes_the_definition_it_replaces():
    """Two `command:` keys in one mapping is not a switch, it is a coin toss."""
    text = (_SCENARIOS / "guides/optimized-baseline.yaml").read_text()
    switched = apply_alternative(text, "sglang")
    assert len(_LAUNCH.findall(switched)) == 1, _LAUNCH.findall(switched)
    assert "sglang.launch_server" in switched.split("command: |", 1)[1]
    assert "vllm serve" not in switched.split("command: |", 1)[1]
