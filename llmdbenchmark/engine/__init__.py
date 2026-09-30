"""Inference-engine support for llm-d-benchmark.

llm-d-benchmark is engine-agnostic: a scenario states the engine's launch
command exactly as it would be typed on a node, and this package reads the few
facts an orchestrator cannot avoid knowing before the process exists -- which
engine, which model, which bind port -- out of that command. No engine flag is
modelled, rendered or validated anywhere else, so supporting a new engine means
adding one :class:`~llmdbenchmark.engine.spec.EngineSpec`.
"""

from llmdbenchmark.engine.alternatives import (
    alternative_engines,
    apply_alternative,
    declared_engines,
    switch_scenario_file,
    tagged_groups,
)
from llmdbenchmark.engine.command import (
    MODEL_READS,
    ParsedCommand,
    parse_command,
    tokenize,
)
from llmdbenchmark.engine.resolver import (
    ENGINE_ROLES,
    compose_command,
    engine_of,
    engine_port,
    model_id_from_commands,
    resolve_engines,
    serving_engine,
    serving_port,
)
from llmdbenchmark.engine.spec import (
    ENGINE_SPECS,
    EngineSpec,
    detect_engine,
    get_engine_spec,
    is_known_engine,
    known_engines,
)

__all__ = [
    "ENGINE_ROLES",
    "ENGINE_SPECS",
    "EngineSpec",
    "MODEL_READS",
    "ParsedCommand",
    "alternative_engines",
    "apply_alternative",
    "compose_command",
    "declared_engines",
    "detect_engine",
    "engine_of",
    "engine_port",
    "get_engine_spec",
    "is_known_engine",
    "known_engines",
    "model_id_from_commands",
    "parse_command",
    "resolve_engines",
    "serving_engine",
    "serving_port",
    "switch_scenario_file",
    "tagged_groups",
    "tokenize",
]
