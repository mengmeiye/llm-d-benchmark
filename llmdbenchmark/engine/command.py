"""Read orchestration facts from a user-supplied engine command.

The command is not rewritten. The parser extracts only the model, port, and
capacity/routing inputs; all other flags stay opaque. Unreadable values produce
notes and fall back to explicit scenario fields.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from llmdbenchmark.engine.spec import (
    EngineSpec,
    detect_engine,
    get_engine_spec,
    is_known_engine,
    launcher_end,
)

# Shell operators that end one command and begin another. Splitting on these
# is how `export FOO=bar; vllm serve ...` yields the serve segment alone.
_SEPARATORS = (";", "&&", "||", "|", "&")

# A leading `VAR=value` assignment (env prefix) on the launch itself.
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

#: Values copied to ``model.*`` for capacity checks and prefix-cache routing.
MODEL_READS = ("maxModelLen", "gpuMemoryUtilization", "blockSize")


@dataclass
class ParsedCommand:
    """Facts inferred from one engine launch command."""

    raw: str
    engine: str | None = None
    argv: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)

    model: str | None = None
    servedModelName: str | None = None
    port: int | None = None

    reads: dict[str, float | int] = field(default_factory=dict)
    preamble: str = ""
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokenizing
# ---------------------------------------------------------------------------


def _strip_continuations(text: str) -> str:
    """Join continued lines and drop whole-line comments."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            lines.append(stripped[:-1])
        else:
            lines.append(stripped + "\n")
    return " ".join("".join(lines).split("\n"))


def tokenize(text: str) -> tuple[list[str], list[str]]:
    """Tokenize shell text, falling back to whitespace splitting on errors."""
    flattened = _strip_continuations(text)
    try:
        lexer = shlex.shlex(flattened, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        return list(lexer), []
    except ValueError as exc:
        return flattened.split(), [f"could not lex command ({exc}); read flags loosely"]


def _split_segments(tokens: list[str]) -> list[list[str]]:
    """Break a token list at shell command separators."""
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(tok)
    return [seg for seg in segments if seg]


# ---------------------------------------------------------------------------
# Flag reading
# ---------------------------------------------------------------------------


def _read_flags(argv: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Read option/value pairs and positionals; repeated options keep the last."""
    flags: dict[str, Any] = {}
    positionals: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-") and tok != "-" and not _looks_negative_number(tok):
            if "=" in tok:
                name, _, value = tok.partition("=")
                flags[name] = value
                i += 1
                continue
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is None or (nxt.startswith("-") and not _looks_negative_number(nxt)):
                flags[tok] = True
                i += 1
            else:
                flags[tok] = nxt
                i += 2
            continue
        positionals.append(tok)
        i += 1
    return flags, positionals


def _looks_negative_number(tok: str) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


def _first(flags: dict[str, Any], names: tuple[str, ...]) -> Any:
    """First present value among ``names`` (spelling aliases of one flag)."""
    for name in names:
        if name in flags:
            return flags[name]
    return None


def _as_int(value: Any) -> int | None:
    if value is None or value is True:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_number(value: Any) -> float | int | None:
    """Read a flag value as int when it is whole, else float."""
    if value is None or value is True:
        return None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_command(text: str | None, engine: str | None = None) -> ParsedCommand:
    """Parse ``text`` using a declared engine or an auto-detected launcher."""
    parsed = ParsedCommand(raw=text or "")
    if not text or not text.strip():
        return parsed

    tokens, notes = tokenize(text)
    parsed.notes.extend(notes)
    if not tokens:
        return parsed

    segments = _split_segments(tokens)

    declared = get_engine_spec(engine) if engine else None
    spec, launch_at, launch_seg = _locate_launch(segments, declared)

    if launch_seg is None:
        # Nothing recognisable: the whole snippet is the launch as far as we
        # can tell. Read flags from the last segment, which is where a launch
        # lives when it follows a preamble.
        launch_seg = segments[-1]
        launch_at = 0
        if spec is None:
            # An unrecognised launcher is not vLLM by default. The generic spec
            # can still read common ``--model`` and ``--port`` spellings while
            # leaving all other flags opaque.
            spec = declared or get_engine_spec("generic")
        if spec.name == "generic":
            parsed.notes.append(
                "could not identify the engine launcher in the command; set "
                "engine.name, engine.image, healthPath and metricsPath "
                "explicitly. A numeric --port is still read; otherwise set "
                "engine.port so the Service and probes have a port to name"
            )
    if (
        spec is not None
        and spec.name == "generic"
        and engine
        and not is_known_engine(engine)
    ):
        # Preserve a custom engine's identity for logs and smoketests. Its
        # operational defaults still come from GENERIC via get_engine_spec().
        parsed.engine = str(engine).strip()
    else:
        parsed.engine = spec.name if spec else (engine or None)

    idx = segments.index(launch_seg)
    if idx > 0:
        parsed.preamble = _rejoin(segments[:idx])

    argv = launch_seg[launch_at:]
    # Drop any leading `VAR=value` env prefix on the launch itself.
    while argv and _ASSIGNMENT.match(argv[0]):
        argv = argv[1:]
    # Drop a subcommand of a launcher that is a command group, so the model
    # positional is read from the right place: `trtllm-serve serve <model>`
    # and `trtllm-serve <model>` name the same invocation.
    if spec is not None and spec.subcommands:
        while argv and argv[0] in spec.subcommands:
            argv = argv[1:]
    parsed.argv = argv

    flags, positionals = _read_flags(argv)
    parsed.flags = flags

    if spec is None:
        return parsed

    # ---- model reference --------------------------------------------------
    model = _first(flags, spec.model_flags)
    if isinstance(model, str):
        parsed.model = model
    elif spec.positional_model and positionals:
        parsed.model = positionals[0]

    served = _first(flags, spec.served_model_flags)
    if isinstance(served, str):
        parsed.servedModelName = served

    # ---- bind port --------------------------------------------------------
    parsed.port = _as_int(_first(flags, spec.port_flags))

    # ---- reads onto model.* -----------------------------------------------
    for metric in MODEL_READS:
        names = spec.flags_for_metric(metric)
        if not names:
            continue
        value = _as_number(_first(flags, names))
        if value is not None:
            parsed.reads[metric] = value

    return parsed


def _locate_launch(
    segments: list[list[str]], declared: EngineSpec | None
) -> tuple[EngineSpec | None, int, list[str] | None]:
    """Find the segment that launches the engine.

    Prefers the declared engine's own launcher signature; falls back to
    detecting any known engine, so a scenario that declares ``sglang`` but
    pastes a ``vllm serve`` command is still read correctly (and the mismatch
    is reported by the resolver).
    """
    if declared is not None:
        for seg in segments:
            end = launcher_end(seg, declared)
            if end is not None:
                return declared, end, seg
    for seg in segments:
        found = detect_engine(seg)
        if found is not None:
            end = launcher_end(seg, found)
            return found, end or 0, seg
    return declared, 0, None


def _rejoin(segments: list[list[str]]) -> str:
    """Re-render preamble segments as a readable shell string (diagnostics only)."""
    return "; ".join(shlex.join(seg) for seg in segments)
