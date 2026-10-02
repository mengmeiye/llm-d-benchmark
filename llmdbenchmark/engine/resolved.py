"""Immutable runtime view of a resolved inference-serving role."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping


LaunchMode = Literal["command", "image-entrypoint"]


def _freeze(value: Any) -> Any:
    """Recursively replace mutable containers with immutable equivalents."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return ordinary containers suitable for YAML/JSON serialization."""
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_thaw(item) for item in value]
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any, default: str = "") -> str:
    return str(value) if value not in (None, "") else default


def _port(value: Any, default: int = 8000) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class ResolvedServingRole:
    """Final serving facts for one role after all input precedence is settled.

    Instances are deeply immutable: nested mappings are read-only and nested
    lists become tuples.  ``to_dict`` deliberately returns fresh mutable
    containers because the result is the on-disk plan representation, not the
    in-process source of truth.
    """

    role: str
    active: bool
    inherited_from: str | None

    engine_name: str
    command: str | None
    launch_mode: LaunchMode
    model_id: str | None
    served_model_name: str | None
    port: int

    image_repository: str
    image_tag: str
    image_pull_policy: str
    container_name: str
    health_path: str
    metrics_path: str
    preprocess_command: str = "/bin/true"
    entrypoint_args: tuple[Any, ...] = ()

    resources: Mapping[str, Any] = field(default_factory=dict)
    parallelism: Mapping[str, Any] = field(default_factory=dict)
    environment: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "resources", _freeze(self.resources))
        object.__setattr__(self, "parallelism", _freeze(self.parallelism))
        object.__setattr__(self, "entrypoint_args", _freeze(self.entrypoint_args))
        object.__setattr__(self, "environment", _freeze(self.environment))

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        role: str,
        *,
        active: bool,
        inherited_from: str | None = None,
        parsed_model: str | None = None,
        parsed_served_model_name: str | None = None,
    ) -> ResolvedServingRole:
        """Build from values already normalized by ``resolve_engines``."""
        role_cfg = _mapping(values.get(role))
        engine_cfg = _mapping(role_cfg.get("engine"))
        common_engine = _mapping(values.get("engine"))
        model_cfg = _mapping(values.get("model"))
        image = _mapping(engine_cfg.get("image"))
        facts = _mapping(engine_cfg.get("facts"))

        command_value = engine_cfg.get("command")
        command = (
            str(command_value)
            if isinstance(command_value, str) and command_value.strip()
            else None
        )
        launch_mode: LaunchMode = "command" if command else "image-entrypoint"

        model_id = (
            _text(parsed_model)
            or _text(facts.get("model"))
            or _text(model_cfg.get("name"))
            or None
        )
        served_model_name = (
            _text(parsed_served_model_name)
            or _text(facts.get("servedModelName"))
            or model_id
            or None
        )
        service_port = common_engine.get("servicePort", 8000)

        args = engine_cfg.get("args")
        environment = role_cfg.get("extraEnvVars")
        return cls(
            role=role,
            active=active,
            inherited_from=inherited_from,
            engine_name=_text(engine_cfg.get("name"), _text(common_engine.get("name"))),
            command=command,
            launch_mode=launch_mode,
            model_id=model_id,
            served_model_name=served_model_name,
            port=_port(engine_cfg.get("port"), _port(service_port)),
            image_repository=_text(image.get("repository")),
            image_tag=_text(image.get("tag")),
            image_pull_policy=_text(
                engine_cfg.get("imagePullPolicy"),
                _text(image.get("pullPolicy"), "IfNotPresent"),
            ),
            container_name=_text(engine_cfg.get("containerName"), "modelserver"),
            health_path=_text(engine_cfg.get("healthPath"), "/health"),
            metrics_path=_text(engine_cfg.get("metricsPath"), "/metrics"),
            preprocess_command=_text(
                engine_cfg.get("preprocessCommand"),
                _text(common_engine.get("preprocessScript"), "/bin/true"),
            ),
            entrypoint_args=tuple(args) if isinstance(args, (list, tuple)) else (),
            resources=_mapping(role_cfg.get("resources")),
            parallelism=_mapping(role_cfg.get("parallelism")),
            environment=(
                tuple(environment) if isinstance(environment, (list, tuple)) else ()
            ),
        )

    @classmethod
    def from_dict(cls, role: str, value: Mapping[str, Any]) -> ResolvedServingRole:
        """Restore a role from a rendered plan."""
        image = _mapping(value.get("image"))
        args = value.get("entrypointArgs")
        environment = value.get("environment")
        inherited = value.get("inheritedFrom")
        command = value.get("command")
        launch_mode = value.get("launchMode")
        if launch_mode not in ("command", "image-entrypoint"):
            launch_mode = "command" if command else "image-entrypoint"

        return cls(
            role=_text(value.get("role"), role),
            active=bool(value.get("active", False)),
            inherited_from=_text(inherited) or None,
            engine_name=_text(value.get("engineName")),
            command=str(command) if isinstance(command, str) and command else None,
            launch_mode=launch_mode,
            model_id=_text(value.get("modelId")) or None,
            served_model_name=_text(value.get("servedModelName")) or None,
            port=_port(value.get("port")),
            image_repository=_text(image.get("repository")),
            image_tag=_text(image.get("tag")),
            image_pull_policy=_text(image.get("pullPolicy"), "IfNotPresent"),
            container_name=_text(value.get("containerName"), "modelserver"),
            health_path=_text(value.get("healthPath"), "/health"),
            metrics_path=_text(value.get("metricsPath"), "/metrics"),
            preprocess_command=_text(value.get("preprocessCommand"), "/bin/true"),
            entrypoint_args=tuple(args) if isinstance(args, (list, tuple)) else (),
            resources=_mapping(value.get("resources")),
            parallelism=_mapping(value.get("parallelism")),
            environment=(
                tuple(environment) if isinstance(environment, (list, tuple)) else ()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable representation stored in rendered plans."""
        return {
            "active": self.active,
            "inheritedFrom": self.inherited_from,
            "engineName": self.engine_name,
            "command": self.command,
            "launchMode": self.launch_mode,
            "modelId": self.model_id,
            "servedModelName": self.served_model_name,
            "port": self.port,
            "image": {
                "repository": self.image_repository,
                "tag": self.image_tag,
                "pullPolicy": self.image_pull_policy,
            },
            "containerName": self.container_name,
            "healthPath": self.health_path,
            "metricsPath": self.metrics_path,
            "preprocessCommand": self.preprocess_command,
            "entrypointArgs": _thaw(self.entrypoint_args),
            "resources": _thaw(self.resources),
            "parallelism": _thaw(self.parallelism),
            "environment": _thaw(self.environment),
        }
