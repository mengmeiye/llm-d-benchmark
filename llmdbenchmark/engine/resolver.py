"""Turn a user-written engine command into the facts the templates need.

This is the whole of llm-d-benchmark's knowledge of an inference engine's
command line, and it is read in two places. :func:`model_id_from_commands`
reads the model id the command names, early, because that id is what
``${model.name}`` substitutes *to*. :func:`resolve_engines` reads everything
else, once, after ``${...}`` substitution.

A scenario states the launch command verbatim. This module reads it and records
what it found under ``<role>.engine``, so the rest of the pipeline consumes
those facts instead of re-deriving engine parameters:

``engine.name``
    The engine actually launched -- detected from the command when the scenario
    does not name one. Selects the default image and the health/metrics paths.
``engine.command``
    The launch line templates render, verbatim. Identical to what the scenario
    wrote unless the role also carries ``engine.extraArgs``, in which case those
    words are appended to it -- see :func:`compose_command`.
``engine.port``
    The port the engine binds, read from the command's ``--port``. The
    container port, the probes and the routing sidecar's upstream all follow
    it, so changing the port in the command changes nothing else. A scenario
    may instead state ``engine.port`` and write ``--port $ENGINE_PORT``.
``engine.healthPath`` / ``engine.metricsPath``
    Where the engine answers. Probes and the PodMonitor read these.
``engine.image``
    Defaulted per engine (``images.vllm``, ``images.sglang``, ...) and
    overridable per role.

What is deliberately *not* here: parallelism widths, accelerator counts, and
every other engine flag. A pod's device count is a Kubernetes fact the kubelet
grants before the process exists, so it is stated in Kubernetes' own vocabulary
(``<role>.resources.limits.<accelerator resource>``, or ``accelerator.count``);
the parallelism the llm-d chart needs for multi-node serving is stated as the
chart value it is. Neither is inferred from the command line.

The only thing here that changes a command is :func:`compose_command`, which
appends ``engine.extraArgs`` to the end of it. Nothing is parsed to do that and
no flag is named: it is concatenation, so a scenario can say "the shared line,
plus these words" without llm-d-benchmark learning what the words mean.

Where a resolved fact contradicts one stated in the scenario, the command wins
and a warning says so -- the command is what the engine will actually do.
"""

from __future__ import annotations

import shlex
from typing import Any

from llmdbenchmark.engine.command import MODEL_READS, ParsedCommand, parse_command
from llmdbenchmark.engine.spec import (
    EngineSpec,
    get_engine_spec,
    is_known_engine,
    known_engines,
)

#: Roles that run an inference engine. Each may carry its own command; a role
#: with none falls back to the top-level ``engine.command``.
#:
#: Order matters: ``standalone`` is resolved after ``decode`` so it can inherit
#: decode's resolved command (see :func:`_inherited_command`).
ENGINE_ROLES: tuple[str, ...] = ("decode", "prefill", "standalone", "nok8s")

#: Which role a role inherits its command from when it states none of its own.
#:
#: Standalone is the same engine serving the same model as decode, with the
#: router taken out from in front of it, so a scenario states the command once.
#: Before this, ``standalone.engine.command`` carried a hardcoded ``vllm serve``
#: default, which meant ``-t standalone`` on any scenario without a standalone
#: block served an engine the scenario never named -- and the three scenarios
#: that did write one had copied decode's line and let the copies drift.
_COMMAND_INHERITS_FROM: dict[str, str] = {"standalone": "decode"}

#: Fragments that make a command specific to a disaggregated pair. A role that
#: inherits one of these is being asked to serve alone with a KV connector and
#: no peer, which is worth saying out loud. Matching is substring-only and the
#: command is never rewritten: stripping flags is exactly the per-engine
#: bookkeeping the verbatim command exists to remove, and only the user knows
#: whether their connector tolerates a missing peer.
_DISAGGREGATION_MARKERS: tuple[str, ...] = (
    "--kv-transfer-config",
    "kv_connector",
    "--kv-events-config",
    "--disaggregation-mode",
    "--disaggregation-bootstrap-port",
)


def resolve_engines(values: dict[str, Any]) -> list[str]:
    """Resolve every engine role in ``values`` in place.

    Runs after ``${...}`` substitution, so a command written with an
    ``${accelerator.*}`` fragment or a ``${namespace.name}`` inside a connector
    payload is read with its real values.

    Returns human-readable warnings. Never raises: an unreadable command still
    renders verbatim, and the warnings tell the user which facts they must state
    explicitly instead.
    """
    warnings: list[str] = []

    common = values.get("engine")
    if not isinstance(common, dict):
        common = {}
        values["engine"] = common

    declared_common = common.get("name")
    if declared_common and not is_known_engine(declared_common):
        warnings.append(
            f"engine.name '{declared_common}' is not a known engine "
            f"({', '.join(known_engines())}); the command will still run "
            "verbatim, but engine.port must be set explicitly because it "
            "cannot be read from it"
        )

    parsed_by_role: dict[str, ParsedCommand] = {}
    for role in ENGINE_ROLES:
        role_cfg = values.get(role)
        if not isinstance(role_cfg, dict):
            continue
        role_warnings, parsed = _resolve_role(values, role, role_cfg, common)
        warnings.extend(role_warnings)
        if parsed is not None:
            parsed_by_role[role] = parsed

    warnings.extend(_check_model_reference(values, parsed_by_role))
    warnings.extend(_adopt_model_reads(values, parsed_by_role))

    return warnings


# ---------------------------------------------------------------------------
# Composing a launch line
# ---------------------------------------------------------------------------


def _shell_word(token: Any) -> str:
    """One ``extraArgs`` entry as a shell word.

    The entries are shell text, the same as the command they join, so a token the
    user has already quoted or written as a ``$VAR`` reference is passed through
    untouched -- quoting it again would turn an expansion into a literal. Anything
    else is quoted only if it would otherwise split or be re-read by the shell,
    which leaves ordinary flags and values exactly as typed.
    """
    text = "" if token is None else str(token)
    if any(ch in text for ch in "$`'\""):
        return text
    return shlex.quote(text)


def compose_command(command: str | None, extra_args: list[Any]) -> str | None:
    """``command`` with ``extra_args`` appended as further words.

    This is the whole mechanism behind ``engine.extraArgs``, and it is
    deliberately concatenation. No token is inspected, so llm-d-benchmark never
    learns which words are flags, which take values, or how a given engine spells
    anything -- the property that keeps a new engine from being a change here.

    Repeating a flag the command already carries is how a value gets overridden,
    and the engine's own argument parser settles it: argparse and click both keep
    the last occurrence. :func:`llmdbenchmark.engine.command._read_flags` reads
    the same way, so what the capacity check and the prefix-cache index see is
    what the engine gets. Nothing is de-duplicated here on purpose -- deciding
    that ``--max-model-len 8192`` should be *removed* when a later
    ``--max-model-len 4096`` appears means knowing that the flag takes a value,
    which is per-flag, per-engine knowledge and exactly the maintenance burden
    this design exists to avoid.

    Appended as a continuation of the last line, indented to match it, so the
    rendered manifest still reads as one command.

    Returns ``command`` unchanged when there is nothing to append, and ``None``
    when there is no command -- ``extraArgs`` has nothing to extend then, which
    the caller reports.
    """
    words = [
        _shell_word(arg) for arg in extra_args if arg is not None and str(arg) != ""
    ]
    if command is None or not words:
        return command

    body = command.rstrip()
    if body.endswith("\\"):
        body = body[:-1].rstrip()

    last = body.rsplit("\n", 1)[-1]
    indent = last[: len(last) - len(last.lstrip())] if "\n" in body else "  "

    return f"{body} \\\n{indent}{' '.join(words)}"


def _extra_args(
    engine_cfg: dict[str, Any],
    common: dict[str, Any],
    role: str,
) -> tuple[list[Any], list[str]]:
    """The words to append to this role's command, and anything wrong with them.

    A role's own ``extraArgs`` replaces the plan-wide ``engine.extraArgs`` rather
    than adding to it, the same way its ``command`` does: one place states the
    words for a role, so there is never a question of what order two lists
    concatenate in.
    """
    warnings: list[str] = []
    for where, source in ((f"{role}.engine", engine_cfg), ("engine", common)):
        value = source.get("extraArgs")
        if value in (None, [], ""):
            continue
        if not isinstance(value, list):
            warnings.append(
                f"{where}.extraArgs must be a list of words (e.g. "
                '["--max-model-len", "4096"]) but is '
                f"{type(value).__name__}; ignoring it, so the command runs "
                "without those flags"
            )
            continue
        return value, warnings
    return [], warnings


# ---------------------------------------------------------------------------
# Inheriting a sibling role's command
# ---------------------------------------------------------------------------


def _inherited_command(
    values: dict[str, Any], role: str, engine_cfg: dict[str, Any]
) -> tuple[str | None, list[str]]:
    """The command ``role`` inherits from its sibling, if it states none.

    Returns ``(command, warnings)``; ``command`` is None when there is nothing
    to inherit, which leaves the role on the image's own entrypoint.

    A role that states ``command: ""`` is opting out explicitly -- it wants the
    image entrypoint -- so it inherits nothing. That is the difference between
    a key written empty and a key left out, and it is the only reason this looks
    at ``engine_cfg`` rather than the already-normalised command.
    """
    source_role = _COMMAND_INHERITS_FROM.get(role)
    if source_role is None:
        return None, []

    stated = engine_cfg.get("command")
    if isinstance(stated, str) and not stated.strip():
        # Explicitly emptied: use the image entrypoint, do not inherit.
        return None, []

    source_cfg = values.get(source_role)
    if not isinstance(source_cfg, dict):
        return None, []
    source_engine = source_cfg.get("engine")
    if not isinstance(source_engine, dict):
        return None, []

    # `source_role` is resolved before `role` (see ENGINE_ROLES), so this is the
    # composed, ${...}-substituted line, not the raw scenario text.
    command = source_engine.get("command")
    if not isinstance(command, str) or not command.strip():
        return None, []

    warnings: list[str] = []
    markers = [m for m in _DISAGGREGATION_MARKERS if m in command]
    if markers:
        warnings.append(
            f"{role} has no engine.command, so it inherits "
            f"{source_role}.engine.command -- which configures disaggregated "
            f"serving ({', '.join(markers)}). Run alone there is no peer to "
            f"transfer with: the engine may refuse to start, or start and stall "
            f"on the first request. Give {role} an engine.command without those "
            f"flags if that is not what you want"
        )
    return command, warnings


# ---------------------------------------------------------------------------
# One role
# ---------------------------------------------------------------------------


def _resolve_role(
    values: dict[str, Any],
    role: str,
    role_cfg: dict[str, Any],
    common: dict[str, Any],
) -> tuple[list[str], ParsedCommand | None]:
    warnings: list[str] = []

    engine_cfg = role_cfg.get("engine")
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}
        role_cfg["engine"] = engine_cfg

    declared = engine_cfg.get("name") or common.get("name")
    command = engine_cfg.get("command")
    if command is None:
        command = common.get("command")
    if command is None:
        command, inherit_warnings = _inherited_command(values, role, engine_cfg)
        warnings.extend(inherit_warnings)
    if isinstance(command, str) and not command.strip():
        command = None

    extra_args, extra_warnings = _extra_args(engine_cfg, common, role)
    warnings.extend(extra_warnings)
    if extra_args and command is None:
        warnings.append(
            f"{role}.engine.extraArgs is set but there is no command to append "
            f"to, so the words are dropped. Give {role} an engine.command, or "
            f"-- if the image's entrypoint launches the server -- pass them as "
            f"{role}.engine.args, which the entrypoint receives"
        )
    command = compose_command(command, extra_args)

    parsed = parse_command(command, declared)
    spec = get_engine_spec(parsed.engine or declared)

    if command is None:
        # No command: the role runs the image's own entrypoint. Legitimate for
        # distroless images that embed their launch (llm-d-inference-sim), and
        # for a role that is simply disabled.
        engine_cfg["command"] = None
        engine_cfg.setdefault("modelCommand", "imageDefault")
    else:
        engine_cfg["command"] = command
        engine_cfg.setdefault("modelCommand", "custom")

    engine_cfg["name"] = spec.name
    engine_cfg["facts"] = parsed.to_dict()

    if (
        declared
        and parsed.engine
        and is_known_engine(declared)
        and get_engine_spec(declared).name != parsed.engine
    ):
        warnings.append(
            f"{role}.engine: scenario declares engine '{declared}' but the "
            f"command launches '{parsed.engine}'; reading the command as "
            f"'{parsed.engine}'"
        )

    for note in parsed.notes:
        warnings.append(f"{role}.engine.command: {note}")

    warnings.extend(_resolve_port(role, engine_cfg, parsed, spec, command))
    _resolve_probe_paths(engine_cfg, spec)
    _resolve_image(values, role, engine_cfg, spec)
    _resolve_container_name(engine_cfg, common, spec)

    return warnings, (parsed if command else None)


def _resolve_port(
    role: str,
    engine_cfg: dict[str, Any],
    parsed: ParsedCommand,
    spec: EngineSpec,
    command: str | None,
) -> list[str]:
    """Settle the port the engine binds inside the container.

    Two spellings, both single-sourced. Either the command carries the number
    (``--port 8200``) and everything else follows it, or the scenario states
    ``engine.port`` and the command references ``--port $ENGINE_PORT``. An
    explicit ``engine.port`` wins, which is also what an image whose entrypoint
    fixes the port needs.
    """
    warnings: list[str] = []
    explicit = engine_cfg.get("port")

    if parsed.port is not None:
        if explicit is not None and int(explicit) != parsed.port:
            warnings.append(
                f"{role}.engine.port is {explicit} but the command binds "
                f"{parsed.port}; using {explicit} for the container port and "
                "probes -- remove one of the two so they cannot drift"
            )
        else:
            engine_cfg["port"] = parsed.port
    elif explicit is None:
        engine_cfg["port"] = spec.defaultPort
        if command:
            warnings.append(
                f"{role}.engine.command has no --port; assuming {spec.name}'s "
                f"default of {spec.defaultPort}. Add --port to the command (or "
                f"set {role}.engine.port) to make it explicit"
            )
    return warnings


def _resolve_probe_paths(engine_cfg: dict[str, Any], spec: EngineSpec) -> None:
    """Publish the engine's health and metrics paths for probes/PodMonitors."""
    engine_cfg.setdefault("healthPath", spec.healthPath)
    engine_cfg.setdefault("metricsPath", spec.metricsPath)


def _resolve_image(
    values: dict[str, Any],
    role: str,
    engine_cfg: dict[str, Any],
    spec: EngineSpec,
) -> None:
    """Default the role's image to the engine's entry under ``images:``.

    A role may pin ``engine.image.{repository,tag,pullPolicy}`` directly; what
    it omits comes from ``images.<engine imageKey>``, which is where the
    per-engine defaults and the version resolver already live.
    """
    images = values.get("images") or {}
    fallback = images.get(spec.imageKey) or images.get("vllm") or {}

    image = engine_cfg.get("image")
    if not isinstance(image, dict):
        image = {}
    for key in ("repository", "tag", "pullPolicy"):
        if image.get(key) in (None, "") and isinstance(fallback, dict):
            if fallback.get(key) not in (None, ""):
                image[key] = fallback[key]
    engine_cfg["image"] = image
    engine_cfg.setdefault("imageKey", spec.imageKey)


def _resolve_container_name(
    engine_cfg: dict[str, Any],
    common: dict[str, Any],
    spec: EngineSpec,
) -> None:
    """Name the serving container.

    Defaults to llm-d's engine-neutral ``modelserver`` so manifests, log
    commands and ``kubectl exec`` lines read the same whichever engine runs.
    Overridable via ``engine.containerName`` for a chart that hardcodes a name.
    """
    name = engine_cfg.get("containerName") or common.get("containerName")
    engine_cfg["containerName"] = name or "modelserver"


# ---------------------------------------------------------------------------
# Facts that belong to the whole plan, not to one role
# ---------------------------------------------------------------------------


def _roles_in_play(values: dict[str, Any], parsed_by_role: dict[str, ParsedCommand]):
    """Parsed commands for roles that are actually going to run.

    A disabled role's command must not decide the plan's model or its capacity
    numbers -- ``prefill`` is present in every scenario and off in most.
    """
    for role, parsed in parsed_by_role.items():
        role_cfg = values.get(role) or {}
        if not isinstance(role_cfg, dict):
            continue
        if role_cfg.get("enabled") is False:
            continue
        try:
            if int(role_cfg.get("replicas", 1)) == 0:
                continue
        except (TypeError, ValueError):
            pass
        yield role, parsed


def _check_model_reference(
    values: dict[str, Any],
    parsed_by_role: dict[str, ParsedCommand],
) -> list[str]:
    """Report a command that serves a different model than the plan does.

    Everything outside the engine -- the model volume, the pod labels, the
    HTTPRoute, the harness target -- is keyed off ``model.name``, so a command
    that serves a different model than the plan states comes up green and
    answers for the wrong weights. Normally ``model.name`` is simply read off
    the command and the two cannot disagree; this fires when something else
    stated a name first (a scenario's own ``model:`` block, a treatment,
    ``-m/--models``). A command that writes the model as ``${model.name}``
    tracks whatever won by construction.

    A ``$VAR`` / ``${...}`` spelling tracks ``model.name`` by construction and
    is not checked.
    """
    warnings: list[str] = []
    model_cfg = values.get("model")
    if not isinstance(model_cfg, dict):
        return warnings

    literal: dict[str, str] = {}
    for role, parsed in _roles_in_play(values, parsed_by_role):
        ref = parsed.servedModelName or parsed.model
        if isinstance(ref, str) and ref and "$" not in ref:
            literal[role] = ref

    if not literal:
        return warnings

    distinct = sorted(set(literal.values()))
    if len(distinct) > 1:
        detail = ", ".join(f"{r}={v}" for r, v in sorted(literal.items()))
        warnings.append(
            "engine commands serve different models in one stack "
            f"({detail}); routing, the model volume and the harness all follow "
            "a single model.name. Split these into separate stacks"
        )
        return warnings

    served = distinct[0]
    current = model_cfg.get("name")
    if current and current != served:
        warnings.append(
            f"model.name is '{current}' but the engine command serves "
            f"'{served}'. Everything outside the engine (model volume path, pod "
            "labels, HTTPRoute, harness target) follows model.name. Serve "
            f"'{current}' in the command, or drop the model.name that "
            f"disagrees so '{served}' is read off the command -- or write the "
            "command's model as `${model.name}` if it has to follow whatever "
            "the plan names"
        )
    return warnings


def model_id_from_commands(values: dict[str, Any]) -> str | None:
    """The model id the engine commands name, when they name one literally.

    A scenario names its model where a user would -- in the launch command,
    either as the serve target or as ``--served-model-name`` -- and does not
    repeat it under ``model:``. Everything outside the engine is keyed off
    ``model.name`` though (the model volume, the pod labels, the HTTPRoute, the
    harness target), so the id is read out of the command instead of typed a
    second time.

    Read from the raw command, before ``${...}`` substitution: a literal id
    needs no substitution, and a ``$``-spelled one already tracks ``model.name``
    by construction and so is skipped here.

    ``engine.extraArgs`` is appended first, exactly as :func:`resolve_engines`
    will append it. The words cannot reach the positional serve target, but they
    can carry a ``--served-model-name``, and the two readers have to agree about
    the id or the later one reports a disagreement with itself.

    Returns ``None`` when no role in play names a literal id, or when two roles
    name different ones -- :func:`resolve_engines` reports that case as the
    misconfiguration it is rather than picking a winner.
    """
    common = values.get("engine")
    if not isinstance(common, dict):
        common = {}

    parsed_by_role: dict[str, ParsedCommand] = {}
    for role in ENGINE_ROLES:
        role_cfg = values.get(role)
        if not isinstance(role_cfg, dict):
            continue
        engine_cfg = role_cfg.get("engine")
        if not isinstance(engine_cfg, dict):
            engine_cfg = {}
        command = engine_cfg.get("command") or common.get("command")
        if not isinstance(command, str) or not command.strip():
            continue
        extra_args, _ = _extra_args(engine_cfg, common, role)
        command = compose_command(command, extra_args)
        declared = engine_cfg.get("name") or common.get("name")
        parsed_by_role[role] = parse_command(command, declared)

    literals = set()
    for _role, parsed in _roles_in_play(values, parsed_by_role):
        ref = parsed.servedModelName or parsed.model
        if isinstance(ref, str) and ref and "$" not in ref:
            literals.add(ref)
    return literals.pop() if len(literals) == 1 else None


def _adopt_model_reads(
    values: dict[str, Any],
    parsed_by_role: dict[str, ParsedCommand],
) -> list[str]:
    """Fill the ``model.*`` numbers named in ``MODEL_READS`` from the commands.

    ``maxModelLen`` and ``gpuMemoryUtilization`` feed the pre-deploy capacity
    check's KV-cache arithmetic and the harness workload profile's context
    length. ``blockSize`` feeds whatever indexes the engine's KV pages -- the
    router's prefix-cache token processor has to hash on the same boundaries, and
    a value that disagrees scores silently against the wrong blocks.

    None of them is rendered into the engine's own manifest: the user already
    wrote each in the flag that sets it, so it is read back rather than restated.
    A scenario states one only when the command cannot -- TRT-LLM has no CLI flag
    for its KV page size, an accelerator overlay knows a kernel picks its own --
    and leaving it unknown turns the consumer off rather than validating against
    a guess.

    When both are present and disagree, the command wins: it is the text the
    engine is handed, so it is the only one of the two that is certainly true,
    and a consumer told the other number would be indexing pages the engine
    never writes. The disagreement is reported either way, because a stated value
    that had to be overruled is a scenario asking for something it is not going
    to get.

    Read values reach the values tree after ``${dotted.path}`` substitution has
    already run once, so a scenario referencing ``${model.blockSize}`` outside a
    command is resolved by the second substitution pass in ``render_plans``.
    """
    warnings: list[str] = []
    model_cfg = values.get("model")
    if not isinstance(model_cfg, dict):
        return warnings

    # Prefer decode, then standalone, then whatever else is in play: decode is
    # the role whose context window the harness drives.
    order = ("decode", "standalone", "nok8s", "prefill")
    in_play = dict(_roles_in_play(values, parsed_by_role))
    for metric in MODEL_READS:
        read_from: str | None = None
        read_value: float | int | None = None
        for role in order:
            parsed = in_play.get(role)
            if parsed is None:
                continue
            value = parsed.reads.get(metric)
            if value is not None:
                read_from, read_value = role, value
                break
        if read_value is None:
            continue

        stated = model_cfg.get(metric)
        model_cfg[metric] = read_value
        if stated in (None, ""):
            continue
        if float(stated) != float(read_value):
            spec = get_engine_spec(in_play[read_from].engine)
            flag = (spec.flags_for_metric(metric) or (metric,))[0]
            warnings.append(
                f"model.{metric} was set to {stated} but {read_from}'s command "
                f"says `{flag} {read_value}`; using {read_value}, which is what "
                f"the engine gets. Drop the model.{metric} to state it once in "
                f"the command, or drop {flag} from the command if {stated} is "
                "the number that holds for this hardware"
            )
    return warnings


# ---------------------------------------------------------------------------
# Shared read helpers (used by templates via the values tree, and by steps)
# ---------------------------------------------------------------------------


def engine_of(values: dict[str, Any], role: str) -> dict[str, Any]:
    """The resolved ``engine`` block for one role (empty dict when absent)."""
    role_cfg = values.get(role) or {}
    engine_cfg = role_cfg.get("engine") if isinstance(role_cfg, dict) else None
    return engine_cfg if isinstance(engine_cfg, dict) else {}


def serving_engine(values: dict[str, Any]) -> dict[str, Any]:
    """The resolved ``engine`` block of the first role that serves.

    A health check dials one endpoint and whatever answers it runs one engine,
    so "the engine of this stack" is well defined for the purpose of naming it
    in a log line and knowing which path to poll. Roles are tried in
    :data:`ENGINE_ROLES` order; the plan-wide ``engine`` block is the fallback
    for a config with no per-role engine at all.
    """
    for role in ENGINE_ROLES:
        cfg = engine_of(values, role)
        if cfg.get("command") or cfg.get("name"):
            return cfg
    top = values.get("engine")
    return top if isinstance(top, dict) else {}


def engine_port(values: dict[str, Any], role: str, default: int = 8000) -> int:
    """The port the engine binds inside ``role``'s container."""
    port = engine_of(values, role).get("port")
    try:
        return int(port)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def serving_port(values: dict[str, Any], default: int | str = 8000) -> int:
    """The port the serving engine actually binds, for dialing a pod directly.

    The command states the port, so the resolved engine block is the only place
    that knows it: ``--port 8200`` in a scenario's serve line lands here as
    ``<role>.engine.port`` and in the container's ``containerPort``. The
    plan-wide ``engine.servicePort`` is the *Service*'s notion of a port and
    defaults to 8000, so reading it to reach a pod IP works only for engines
    that happen to bind 8000 -- an sglang command on 8200 is then probed on a
    port nothing listens to.

    Falls back to ``engine.servicePort`` and then to ``default`` for a config
    with no resolved engine at all (``--dry-run``, a bare values tree).
    """
    for candidate in (
        serving_engine(values).get("port"),
        (values.get("engine") or {}).get("servicePort"),
        default,
    ):
        try:
            return int(candidate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return 8000
