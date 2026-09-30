"""Engines a scenario can be switched to without editing it.

A scenario states its engine as a verbatim launch command, so "run this same
stack on another engine" is an edit to one string -- and a scenario file will
often carry that other string already, commented out, next to the prose
explaining it. Commented text is covered by nothing, though: it can drift out of
step with the engine's flags, with the file's own indentation, or with a
companion key that has to move alongside it, and nothing notices until someone
uncomments it by hand.

Tagging each commented group with ``# @engine <name>`` makes the switch
machine-readable. :func:`alternative_engines` reports which engines a file
offers and :func:`apply_alternative` performs the edit, so the same switch can
be rendered by a test, planned by ``util/scenario-inventory.py --apply`` and run
by ``llmdbenchmark --engine <name>``.

The edit is not a plain uncomment. Each group *replaces* a definition that is
live in the file, and leaving that definition in place produces either invalid
YAML (an uncommented ``extraEnvVars:`` list under a live ``extraEnvVars: []``)
or a mapping with two ``command:`` keys, which is not a switch but a coin toss.
So the replaced keys are deleted as part of applying the group.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from llmdbenchmark.engine.command import parse_command

#: Roles that carry a launch command, so a scenario's declared engines can be
#: read without resolving the whole values tree.
_COMMAND_ROLES = ("decode", "prefill", "standalone", "nok8s")

#: Tags the commented-out group directly below it as belonging to one engine.
ALT_TAG = re.compile(r"^(\s*)#\s*@engine\s+(\S+)\s*$")

#: Strips one comment marker, and only one: a nested line such as
#: ``#   metricsPath: x`` must keep the indentation that follows the marker.
UNCOMMENT = re.compile(r"^(\s*)# ?")

#: A mapping key at the start of a line.
KEY = re.compile(r"^(\s*)([\w.\-/]+):")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_code(line: str) -> bool:
    """A line YAML reads: not blank, not a comment."""
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def tagged_groups(text: str) -> list[tuple[str, int, int]]:
    """``(engine, tag_line, end)`` for each ``# @engine NAME`` group in ``text``.

    The group is the run of commented lines directly below the tag, ending at
    the first blank or uncommented line -- the same thing a reader sees as "the
    block below the tag". A tag with nothing commented under it is not a group,
    which is what keeps prose *quoting* the tag from being mistaken for one.
    """
    lines = text.splitlines()
    groups = []
    for index, line in enumerate(lines):
        tag = ALT_TAG.match(line)
        if not tag:
            continue
        end = index + 1
        while end < len(lines) and lines[end].strip().startswith("#"):
            end += 1
        if end > index + 1:
            groups.append((tag.group(2).lower(), index, end))
    return groups


def alternative_engines(text: str) -> list[str]:
    """Engines ``text`` carries a commented-out alternative for."""
    return sorted({engine for engine, _, _ in tagged_groups(text)})


def declared_engines(text: str) -> list[str]:
    """Engines the *live* commands in ``text`` launch.

    Read off each role's ``engine.command`` by launcher signature, the same way
    the renderer reads it. Used only to tell "switch this scenario" apart from
    "this scenario is already that engine", so an unparseable file or a command
    that names no known launcher simply reports nothing.
    """
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        for role in _COMMAND_ROLES:
            role_cfg = node.get(role)
            engine_cfg = role_cfg.get("engine") if isinstance(role_cfg, dict) else None
            command = (
                engine_cfg.get("command") if isinstance(engine_cfg, dict) else None
            )
            if isinstance(command, str) and command.strip():
                engine = parse_command(command).engine
                if engine:
                    found.add(engine)
        for value in node.values():
            walk(value)

    walk(document)
    return sorted(found)


def _scope(lines: list[str], at: int, indent: int) -> tuple[int, int]:
    """Line range around ``at`` that stays inside the mapping ``indent`` is in.

    Walks out in both directions until a line YAML reads is shallower than
    ``indent`` -- i.e. until the enclosing mapping ends. Comments and blanks
    never close a mapping, so they are stepped over, which is what lets a
    commented group sit under a paragraph of prose and still belong to the key
    above it.
    """
    low = at
    while low > 0 and not (
        _is_code(lines[low - 1]) and _indent(lines[low - 1]) < indent
    ):
        low -= 1
    high = at
    while high < len(lines) and not (
        _is_code(lines[high]) and _indent(lines[high]) < indent
    ):
        high += 1
    return low, high


def _block_end(lines: list[str], start: int) -> int:
    """End of the value that begins on line ``start``.

    Everything more deeply indented belongs to it, as do same-indent sequence
    items (a list written flush with its key) and any blank lines between.
    """
    indent = _indent(lines[start])
    end = last = start + 1
    while end < len(lines):
        line = lines[end]
        if not line.strip():
            end += 1
            continue
        if _indent(line) > indent or (
            _indent(line) == indent and line.lstrip().startswith("-")
        ):
            end += 1
            last = end
            continue
        break
    return last


def apply_alternative(text: str, engine: str) -> str:
    """``text`` with every ``# @engine <engine>`` group uncommented.

    This is the edit the scenario's own comments ask the reader to make, done
    mechanically so it can be tested: strip one comment marker from each line of
    the tagged groups, drop the tag lines, and delete the active definition each
    group replaces.

    Groups are applied bottom-up so that a deletion never moves a group that
    has not been applied yet.
    """
    groups = [g for g in tagged_groups(text) if g[0] == engine.lower()]
    if not groups:
        raise ValueError(
            f"no `# @engine {engine}` group here; "
            f"this file offers {alternative_engines(text) or 'none'}"
        )
    lines = text.splitlines()
    for _, tag, end in sorted(groups, reverse=True):
        body = [UNCOMMENT.sub(r"\1", line, count=1) for line in lines[tag + 1 : end]]
        base = min((_indent(line) for line in body if line.strip()), default=0)
        replaced = [
            match.group(2)
            for match in (KEY.match(line) for line in body)
            if match and len(match.group(1)) == base
        ]
        low, high = _scope(lines, tag, base)
        for key in replaced:
            for index in range(low, high):
                if index in range(tag, end) or not _is_code(lines[index]):
                    continue
                match = KEY.match(lines[index])
                if match and match.group(2) == key and len(match.group(1)) == base:
                    stop = _block_end(lines, index)
                    del lines[index:stop]
                    if index < tag:
                        shift = stop - index
                        tag, end, low, high = (
                            tag - shift,
                            end - shift,
                            low,
                            high - shift,
                        )
                    break
        lines[tag:end] = body
    return "\n".join(lines) + "\n"


def switch_scenario_file(scenario: Path, engine: str, out_dir: Path) -> Path | None:
    """Write ``scenario`` switched to ``engine``, or None if it already is.

    The repo is never touched: the switched copy is written under ``out_dir``
    (the run's plan directory, so it is kept with the run's other artifacts) and
    its path returned for the caller to render instead of the original.

    Returns None when ``scenario`` already launches ``engine`` and offers no
    alternative for it -- asking for the engine a file already states is a
    no-op, not an error. Raises :class:`ValueError` when the file neither states
    nor offers it.
    """
    scenario = Path(scenario)
    text = scenario.read_text(encoding="utf-8")
    wanted = engine.strip().lower()
    if wanted not in alternative_engines(text):
        if wanted in declared_engines(text):
            return None
        offered = alternative_engines(text)
        raise ValueError(
            f"{scenario} cannot be switched to '{engine}': it launches "
            f"{declared_engines(text) or 'no recognised engine'} and offers "
            f"{offered if offered else 'no alternative'}. An alternative is a "
            f"commented-out group tagged `# @engine {engine}`."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    switched = out_dir / f"{scenario.stem}-{wanted}.yaml"
    switched.write_text(apply_alternative(text, wanted), encoding="utf-8")
    return switched
