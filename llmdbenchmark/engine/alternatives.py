"""Apply ``# @engine NAME`` alternative blocks from scenario files."""

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
    """Return ``(engine, tag line, end line)`` for tagged comment blocks."""
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
    """Return engines detected in live role commands."""
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
    """Return the enclosing YAML mapping's line range."""
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
    """Return the line after the YAML value beginning at ``start``."""
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
    """Uncomment an engine's groups and remove the definitions they replace."""
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
    """Write a switched copy under ``out_dir``; return ``None`` if unchanged."""
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
