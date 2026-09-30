"""Minimal reader for a user-supplied engine launch command.

The scenario carries the engine command **verbatim** -- whatever the user would
type on a node, copied in unchanged:

.. code-block:: yaml

   decode:
     engine:
       command: |
         vllm serve Qwen/Qwen3-32B \
           --port 8200 \
           --tensor-parallel-size 4 \
           --gpu-memory-utilization 0.95

Nothing in this module rewrites that text. It only *reads* it, and only for
what cannot wait until the process exists:

* the model the command serves, because the model volume and the routing target
  are created before the pod is;
* the port it binds, because the Service and the probes have to name one;
* the context length and the memory fraction, because the pre-deploy capacity
  check sizes KV cache against them;
* the KV page size, because the router's prefix-cache index has to hash on the
  same block boundaries the engine writes.

Every other flag is opaque and reaches the container untouched, which is the
point: there is no table of engine parameters to keep up to date.

Reading is deliberately forgiving. A snippet we cannot read is not an error --
:func:`parse_command` records what it could not determine in ``notes`` and the
caller falls back to an explicit scenario field. A malformed command is the
engine's to report, where its own error message is far more useful than ours.
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
    launcher_end,
)

# Shell operators that end one command and begin another. Splitting on these
# is how `export FOO=bar; vllm serve ...` yields the serve segment alone.
_SEPARATORS = (";", "&&", "||", "|", "&")

# A leading `VAR=value` assignment (env prefix) on the launch itself.
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

#: Numbers that something outside the engine needs and the command already
#: states (see ``EngineSpec.flags_for_metric``). Reported under
#: :attr:`ParsedCommand.reads` and copied onto ``model.*`` by the resolver, so
#: no scenario states them twice:
#:
#:   maxModelLen, gpuMemoryUtilization  the pre-deploy capacity check, and the
#:                                      harness workload profile's context length
#:   blockSize                          the router's prefix-cache index, which
#:                                      must hash on the engine's page boundaries
MODEL_READS = ("maxModelLen", "gpuMemoryUtilization", "blockSize")


@dataclass
class ParsedCommand:
    """Facts read out of one engine launch command.

    ``raw`` is the authoritative text; the template renders it and nothing
    else. Every other attribute is something we inferred so that the Service,
    the probes, the capacity check and the prefix-cache index can be sized
    without the user restating what the command already says.
    """

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

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view, published into the values tree as ``.facts``."""
        return {
            "engine": self.engine,
            "model": self.model,
            "servedModelName": self.servedModelName,
            "port": self.port,
            "reads": dict(self.reads),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Tokenizing
# ---------------------------------------------------------------------------


def _strip_continuations(text: str) -> str:
    """Join backslash-continued lines and drop whole-line comments.

    A pasted command is almost always multi-line with trailing ``\\``. Both the
    backslash form and a bare newline inside one logical command have to
    collapse to whitespace before :mod:`shlex` sees them, because shlex in
    POSIX mode treats a lone ``\\`` as an escape of the newline character
    rather than a line join.
    """
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
    """Split a shell snippet into tokens.

    Returns ``(tokens, notes)``. Quoting is honoured (so a single-quoted JSON
    blob stays one token) and quotes are stripped, which is what we want for
    reading a value. On a lexing failure we degrade to whitespace splitting
    rather than giving up -- the facts may still be readable, and the verbatim
    text is unaffected either way.
    """
    flattened = _strip_continuations(text)
    try:
        return shlex.split(flattened, comments=False, posix=True), []
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
    """Read ``--flag value`` / ``--flag=value`` / bare ``--flag`` pairs.

    Returns ``(flags, positionals)``. A repeated flag keeps the last value,
    matching how argparse and click both behave. A bare flag maps to ``True``;
    a flag whose next token is another flag is treated as bare, which is the
    only ambiguity a reader without the engine's own argument table can hit,
    and it is the correct reading for every store_true flag.
    """
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
    """Read the facts out of an engine launch command.

    Parameters
    ----------
    text:
        The command exactly as the user wrote it. May span lines, carry a
        preamble (``export``, ``source``, ``mkdir``) and use any quoting.
    engine:
        The engine declared in the scenario, when there is one. Used as the
        spec to read flags with; when omitted the engine is detected from the
        launcher token itself.
    """
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
            spec = declared or get_engine_spec(engine)
            parsed.notes.append(
                "could not identify the engine launcher in the command; set "
                "engine.port in the scenario (or write --port $ENGINE_PORT) so "
                "the Service and the probes have a port to name"
            )
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
    model = _first(flags, spec.modelFlags)
    if isinstance(model, str):
        parsed.model = model
    elif spec.positionalModel and positionals:
        parsed.model = positionals[0]

    served = _first(flags, spec.servedModelFlags)
    if isinstance(served, str):
        parsed.servedModelName = served

    # ---- bind port --------------------------------------------------------
    parsed.port = _as_int(_first(flags, spec.portFlags))

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
