"""Resolve engine commands into the runtime facts templates and steps consume.

``model_id_from_commands`` runs before config-variable substitution so a model
named only in a command can seed ``model.*``. ``resolve_engines`` then resolves
each role once and publishes immutable ``resolvedServingRoles`` snapshots.
Parallelism and accelerator allocation remain explicit Kubernetes/chart data;
they are not inferred from engine flags.
"""

from __future__ import annotations

import shlex
from typing import Any

from llmdbenchmark.engine.command import MODEL_READS, ParsedCommand, parse_command
from llmdbenchmark.engine.resolved import ResolvedServingRole
from llmdbenchmark.engine.spec import (
    EngineSpec,
    get_engine_spec,
    is_known_engine,
    known_engines,
)

#: Order matters because standalone may inherit decode's resolved command.
ENGINE_ROLES: tuple[str, ...] = ("decode", "prefill", "standalone", "nok8s")

#: Serialized home of the immutable role snapshots in a rendered plan.
RESOLVED_SERVING_ROLES_KEY = "resolvedServingRoles"

#: A standalone deployment reuses decode's command unless explicitly overridden.
_COMMAND_INHERITS_FROM: dict[str, str] = {"standalone": "decode"}

# Settings that describe how an inherited command runs. These follow the
# command unless the target role states its own value.
_COMMAND_RUNTIME_FIELDS: tuple[str, ...] = (
    "name",
    "port",
    "preprocessCommand",
    "image",
    "healthPath",
    "metricsPath",
    "containerName",
)

#: Markers that warrant a warning when a disaggregated command is inherited by
#: standalone. The command is never rewritten.
_DISAGGREGATION_MARKERS: tuple[str, ...] = (
    "--kv-transfer-config",
    "kv_connector",
    "--kv-events-config",
    "--disaggregation-mode",
    "--disaggregation-bootstrap-port",
)


def resolve_engines(values: dict[str, Any]) -> list[str]:
    """Resolve all engine roles in place and return non-fatal warnings."""
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
            "verbatim. Set each role's engine.image, healthPath and "
            "metricsPath explicitly, and engine.port too unless the command "
            "contains a numeric --port"
        )

    parsed_by_role: dict[str, ParsedCommand] = {}
    inherited_by_role: dict[str, str] = {}
    for role in ENGINE_ROLES:
        role_cfg = values.get(role)
        if not isinstance(role_cfg, dict):
            continue
        role_warnings, parsed, inherited_from = _resolve_role(
            values, role, role_cfg, common
        )
        warnings.extend(role_warnings)
        if parsed is not None:
            parsed_by_role[role] = parsed
        if inherited_from is not None:
            inherited_by_role[role] = inherited_from

    warnings.extend(_check_model_reference(values, parsed_by_role))
    warnings.extend(_adopt_model_reads(values, parsed_by_role))
    _publish_serving_roles(values, inherited_by_role, parsed_by_role)

    return warnings


# ---------------------------------------------------------------------------
# Composing a launch line
# ---------------------------------------------------------------------------


def _shell_word(token: Any) -> str:
    """Quote one extra argument unless it already contains shell syntax."""
    text = "" if token is None else str(token)
    if any(ch in text for ch in "$`'\""):
        return text
    return shlex.quote(text)


def compose_command(command: str | None, extra_args: list[Any]) -> str | None:
    """Append opaque shell words without interpreting or deduplicating flags."""
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
    *,
    include_common: bool = True,
) -> tuple[list[Any], list[str]]:
    """Return role-level or plan-wide extra arguments and validation warnings."""
    warnings: list[str] = []
    sources = [(f"{role}.engine", engine_cfg)]
    if include_common:
        sources.append(("engine", common))
    for where, source in sources:
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
    """Return a sibling command to inherit; an explicit empty command opts out."""
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
    if markers and _role_is_active(values.get(role)):
        warnings.append(
            f"{role} has no engine.command, so it inherits "
            f"{source_role}.engine.command -- which configures disaggregated "
            f"serving ({', '.join(markers)}). Run alone there is no peer to "
            f"transfer with: the engine may refuse to start, or start and stall "
            f"on the first request. Give {role} an engine.command without those "
            f"flags if that is not what you want"
        )
    return command, warnings


def _inherit_command_runtime(
    values: dict[str, Any], role: str, engine_cfg: dict[str, Any]
) -> None:
    """Copy runtime settings coupled to an inherited sibling command."""
    source_role = _COMMAND_INHERITS_FROM.get(role)
    source_cfg = values.get(source_role) if source_role else None
    source_engine = source_cfg.get("engine") if isinstance(source_cfg, dict) else None
    if not isinstance(source_engine, dict):
        return

    for field in _COMMAND_RUNTIME_FIELDS:
        current = engine_cfg.get(field)
        if current not in (None, "", {}):
            continue
        source_value = source_engine.get(field)
        if source_value in (None, "", {}):
            continue
        engine_cfg[field] = (
            dict(source_value) if isinstance(source_value, dict) else source_value
        )


# ---------------------------------------------------------------------------
# One role
# ---------------------------------------------------------------------------


def _resolve_role(
    values: dict[str, Any],
    role: str,
    role_cfg: dict[str, Any],
    common: dict[str, Any],
) -> tuple[list[str], ParsedCommand | None, str | None]:
    warnings: list[str] = []

    engine_cfg = role_cfg.get("engine")
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}
        role_cfg["engine"] = engine_cfg

    command = engine_cfg.get("command")
    inherited = False
    if command is None:
        command = common.get("command")
    if command is None:
        command, inherit_warnings = _inherited_command(values, role, engine_cfg)
        warnings.extend(inherit_warnings)
        inherited = command is not None
        if inherited:
            _inherit_command_runtime(values, role, engine_cfg)
    if isinstance(command, str) and not command.strip():
        command = None

    # A sibling's resolved command already includes the plan-wide extraArgs.
    # Only this role's own list may extend it further; applying the common list
    # again can duplicate flags whose parsers accumulate repeated values.
    extra_args, extra_warnings = _extra_args(
        engine_cfg, common, role, include_common=not inherited
    )
    warnings.extend(extra_warnings)
    if extra_args and command is None:
        warnings.append(
            f"{role}.engine.extraArgs is set but there is no command to append "
            f"to, so the words are dropped. Give {role} an engine.command, or "
            f"-- if the image's entrypoint launches the server -- pass them as "
            f"{role}.engine.args, which the entrypoint receives"
        )
    command = compose_command(command, extra_args)

    declared = engine_cfg.get("name") or common.get("name")
    parsed = parse_command(command, declared)
    spec = get_engine_spec(parsed.engine or declared)

    if command is None:
        # No command: the role runs the image's own entrypoint. Legitimate for
        # distroless images that embed their launch (llm-d-inference-sim), and
        # for a role that is simply disabled.
        engine_cfg["command"] = None
    else:
        engine_cfg["command"] = command

    # A declared custom name is useful in logs and smoketest output even though
    # its operational defaults come from the generic spec.
    engine_cfg["name"] = parsed.engine or spec.name

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
    _resolve_image(values, engine_cfg, spec)
    _resolve_container_name(engine_cfg, common)

    inherited_from = _COMMAND_INHERITS_FROM.get(role) if inherited else None
    return warnings, (parsed if command else None), inherited_from


def _resolve_port(
    role: str,
    engine_cfg: dict[str, Any],
    parsed: ParsedCommand,
    spec: EngineSpec,
    command: str | None,
) -> list[str]:
    """Resolve the bind port; a numeric command option is authoritative."""
    warnings: list[str] = []
    explicit = engine_cfg.get("port")

    if parsed.port is not None:
        try:
            explicit_port = int(explicit) if explicit is not None else None
        except (TypeError, ValueError):
            explicit_port = None
        if explicit is not None and explicit_port != parsed.port:
            warnings.append(
                f"{role}.engine.port is {explicit} but the command binds "
                f"{parsed.port}; using {parsed.port} for the container port and "
                "probes because the command is authoritative"
            )
        engine_cfg["port"] = parsed.port
    elif explicit is None:
        engine_cfg["port"] = spec.default_port
        if command:
            warnings.append(
                f"{role}.engine.command has no --port; assuming {spec.name}'s "
                f"default of {spec.default_port}. Add --port to the command (or "
                f"set {role}.engine.port) to make it explicit"
            )
    return warnings


def _resolve_probe_paths(engine_cfg: dict[str, Any], spec: EngineSpec) -> None:
    """Publish the engine's health and metrics paths for probes/PodMonitors."""
    engine_cfg.setdefault("healthPath", spec.health_path)
    engine_cfg.setdefault("metricsPath", spec.metrics_path)


def _resolve_image(
    values: dict[str, Any],
    engine_cfg: dict[str, Any],
    spec: EngineSpec,
) -> None:
    """Default the role's image to the engine's entry under ``images:``.

    A role may pin ``engine.image.{repository,tag,pullPolicy}`` directly; what
    it omits comes from ``images.<engine imageKey>``, which is where the
    per-engine defaults and the version resolver already live.
    """
    images = values.get("images") or {}
    fallback = images.get(spec.image_key) or images.get("vllm") or {}

    image = engine_cfg.get("image")
    if not isinstance(image, dict):
        image = {}
    for key in ("repository", "tag", "pullPolicy"):
        if image.get(key) in (None, "") and isinstance(fallback, dict):
            if fallback.get(key) not in (None, ""):
                image[key] = fallback[key]
    engine_cfg["image"] = image


def _resolve_container_name(
    engine_cfg: dict[str, Any],
    common: dict[str, Any],
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


def _role_is_active(role_cfg: Any) -> bool:
    """Whether a role's process can run in the rendered deployment."""
    if not isinstance(role_cfg, dict) or role_cfg.get("enabled") is False:
        return False
    try:
        return int(role_cfg.get("replicas", 1)) != 0
    except (TypeError, ValueError):
        return True


def _serving_role_order(values: dict[str, Any]) -> tuple[str, ...]:
    """Engine roles reachable through the selected deployment method."""
    modelservice = values.get("modelservice")
    standalone = values.get("standalone")
    nok8s = values.get("nok8s")

    if isinstance(modelservice, dict) and modelservice.get("enabled") is True:
        return ("decode", "prefill")
    if isinstance(standalone, dict) and standalone.get("enabled") is True:
        return ("standalone",)
    if isinstance(nok8s, dict) and nok8s.get("enabled") is True:
        return ("nok8s",)
    # Bare/unresolved values trees used by callers and unit tests may not carry
    # method flags yet. Preserve the general role fallback for them.
    return ENGINE_ROLES


def _roles_in_play(values: dict[str, Any], parsed_by_role: dict[str, ParsedCommand]):
    """Parsed commands for roles that are actually going to run.

    A disabled role's command must not decide the plan's model or its capacity
    numbers -- ``prefill`` is present in every scenario and off in most.
    """
    eligible = set(_serving_role_order(values))
    for role, parsed in parsed_by_role.items():
        if role in eligible and _role_is_active(values.get(role)):
            yield role, parsed


def _check_model_reference(
    values: dict[str, Any],
    parsed_by_role: dict[str, ParsedCommand],
) -> list[str]:
    """Warn when active commands and ``model.name`` identify different models."""
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
    """Return the single literal model id named by active commands, if any.

    This early pass mirrors command inheritance and ``extraArgs`` handling so
    the id can seed ``model.*`` before config-variable substitution.
    """
    common = values.get("engine")
    if not isinstance(common, dict):
        common = {}

    parsed_by_role: dict[str, ParsedCommand] = {}
    composed_by_role: dict[str, str] = {}
    for role in ENGINE_ROLES:
        role_cfg = values.get(role)
        if not isinstance(role_cfg, dict):
            continue
        engine_cfg = role_cfg.get("engine")
        if not isinstance(engine_cfg, dict):
            engine_cfg = {}

        # Follow the same precedence as `_resolve_role`: a role command, then
        # the plan-wide command, then a sibling's already-composed command.
        # The final case is what lets a standalone render discover the model
        # from decode before `resolve_engines` materialises that inheritance in
        # the values tree. An explicitly empty role command opts into the
        # image's entrypoint and therefore must not inherit.
        command = engine_cfg.get("command")
        inherited = False
        if command is None:
            command = common.get("command")
        if command is None:
            source_role = _COMMAND_INHERITS_FROM.get(role)
            if source_role is not None:
                command = composed_by_role.get(source_role)
                inherited = command is not None
        if not isinstance(command, str) or not command.strip():
            continue
        extra_args, _ = _extra_args(
            engine_cfg, common, role, include_common=not inherited
        )
        command = compose_command(command, extra_args)
        if command is None:
            continue
        composed_by_role[role] = command
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
    """Copy capacity/routing inputs from active commands onto ``model.*``.

    A command value wins over a conflicting fallback because it is what the
    engine receives; the conflict is returned as a warning.
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
# Immutable serving-role snapshots and compatibility helpers
# ---------------------------------------------------------------------------


def _build_serving_roles(
    values: dict[str, Any],
    inherited_by_role: dict[str, str] | None = None,
    parsed_by_role: dict[str, ParsedCommand] | None = None,
) -> dict[str, ResolvedServingRole]:
    """Build role snapshots without parsing commands or mutating ``values``."""
    eligible = set(_serving_role_order(values))
    inherited_by_role = inherited_by_role or {}
    parsed_by_role = parsed_by_role or {}
    roles: dict[str, ResolvedServingRole] = {}
    for role in ENGINE_ROLES:
        if not isinstance(values.get(role), dict):
            continue
        parsed = parsed_by_role.get(role)
        roles[role] = ResolvedServingRole.from_values(
            values,
            role,
            active=role in eligible and _role_is_active(values.get(role)),
            inherited_from=inherited_by_role.get(role),
            parsed_model=parsed.model if parsed is not None else None,
            parsed_served_model_name=(
                parsed.servedModelName if parsed is not None else None
            ),
        )
    return roles


def _publish_serving_roles(
    values: dict[str, Any],
    inherited_by_role: dict[str, str] | None = None,
    parsed_by_role: dict[str, ParsedCommand] | None = None,
) -> dict[str, ResolvedServingRole]:
    """Build immutable roles and publish their serializable plan representation."""
    roles = _build_serving_roles(values, inherited_by_role, parsed_by_role)
    values[RESOLVED_SERVING_ROLES_KEY] = {
        role: resolved.to_dict() for role, resolved in roles.items()
    }
    return roles


def resolved_serving_roles(
    values: dict[str, Any],
) -> dict[str, ResolvedServingRole]:
    """Read plan snapshots, deriving them without reparsing for legacy plans."""
    serialized = values.get(RESOLVED_SERVING_ROLES_KEY)
    if isinstance(serialized, dict):
        roles = {
            str(role): ResolvedServingRole.from_dict(str(role), value)
            for role, value in serialized.items()
            if isinstance(value, dict)
        }
        if roles:
            return roles
    return _build_serving_roles(values)


def resolved_serving_role(
    values: dict[str, Any], role: str
) -> ResolvedServingRole | None:
    """Return one resolved role, or ``None`` when the role is absent."""
    return resolved_serving_roles(values).get(role)


def serving_role(values: dict[str, Any]) -> ResolvedServingRole | None:
    """Return the active role behind the stack's inference endpoint."""
    roles = resolved_serving_roles(values)
    for role in _serving_role_order(values):
        resolved = roles.get(role)
        if resolved is not None and resolved.active:
            return resolved
    return None


def serving_port(values: dict[str, Any], default: int | str = 8000) -> int:
    """Return the active engine's bind port, with legacy-plan fallbacks."""
    resolved = serving_role(values)
    for candidate in (
        resolved.port if resolved is not None else None,
        (values.get("engine") or {}).get("servicePort"),
        default,
    ):
        try:
            return int(candidate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return 8000
