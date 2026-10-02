"""Minimal registry for facts orchestration must read from launch commands.

Each spec identifies a server launcher, its model and port options, optional
capacity inputs, and endpoint/image defaults. All other engine arguments remain
opaque. Flag spellings are sourced from the local vLLM, SGLang, TensorRT-LLM,
and llm-d-inference-sim implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

#: Denominators used by each engine's memory-fraction option. The capacity
#: validator distinguishes whole-device (vLLM), weights-plus-KV (SGLang), and
#: free-after-load (TensorRT-LLM) fractions.
MEMORY_FRACTION_DEVICE = "device"
MEMORY_FRACTION_WEIGHTS_AND_KV = "weightsAndKv"
MEMORY_FRACTION_FREE_AFTER_LOAD = "freeAfterLoad"

MEMORY_FRACTION_SCOPES: tuple[str, ...] = (
    MEMORY_FRACTION_DEVICE,
    MEMORY_FRACTION_WEIGHTS_AND_KV,
    MEMORY_FRACTION_FREE_AFTER_LOAD,
)


@dataclass(frozen=True, slots=True)
class EngineSpec:
    """What the orchestrator must know about one inference engine."""

    name: str

    # Token sequences that identify a server invocation.
    launchers: tuple[tuple[str, ...], ...] = ()

    # Command-group words skipped before positional arguments are read.
    subcommands: tuple[str, ...] = ()

    # Model and served-name options. positionalModel accepts a bare model after
    # the launcher.
    model_flags: tuple[str, ...] = ()
    positional_model: bool = False
    served_model_flags: tuple[str, ...] = ()

    port_flags: tuple[str, ...] = ("--port",)

    # Optional values consumed by capacity checks and prefix-cache routing.
    max_model_len_flags: tuple[str, ...] = ()
    memory_util_flags: tuple[str, ...] = ()
    block_size_flags: tuple[str, ...] = ()

    memory_fraction_scope: str = MEMORY_FRACTION_DEVICE

    default_port: int = 8000
    health_path: str = "/health"
    metrics_path: str = "/metrics"

    image_key: str = "vllm"

    aliases: tuple[str, ...] = ()

    def flags_for_metric(self, metric: str) -> tuple[str, ...]:
        """Return the flag spellings for one ``MODEL_READS`` entry."""
        return {
            "maxModelLen": self.max_model_len_flags,
            "gpuMemoryUtilization": self.memory_util_flags,
            "blockSize": self.block_size_flags,
        }.get(metric, ())


VLLM = EngineSpec(
    name="vllm",
    launchers=(
        ("vllm", "serve"),
        ("-m", "vllm.entrypoints.openai.api_server"),
    ),
    model_flags=("--model",),
    positional_model=True,
    served_model_flags=("--served-model-name",),
    max_model_len_flags=("--max-model-len",),
    memory_util_flags=("--gpu-memory-utilization",),
    memory_fraction_scope=MEMORY_FRACTION_DEVICE,
    block_size_flags=("--block-size",),
    default_port=8000,
    health_path="/health",
    metrics_path="/metrics",
    image_key="vllm",
)

SGLANG = EngineSpec(
    name="sglang",
    launchers=(
        ("sglang", "serve"),
        ("-m", "sglang.launch_server"),
        ("sglang.launch_server",),
    ),
    model_flags=("--model-path", "--model"),
    positional_model=True,
    served_model_flags=("--served-model-name",),
    max_model_len_flags=("--context-length",),
    memory_util_flags=("--mem-fraction-static",),
    memory_fraction_scope=MEMORY_FRACTION_WEIGHTS_AND_KV,
    block_size_flags=("--page-size",),
    default_port=30000,
    health_path="/health",
    metrics_path="/metrics",
    image_key="sglang",
)

TRTLLM = EngineSpec(
    name="trtllm",
    launchers=(("trtllm-serve",),),
    subcommands=(
        "serve",
        "disaggregated",
        "disaggregated_mpi_worker",
        "mm_embedding_serve",
        "embeddings",
    ),
    model_flags=("--model", "--model_path"),
    positional_model=True,
    served_model_flags=("--served_model_name", "--served-model-name"),
    max_model_len_flags=("--max_seq_len",),
    memory_util_flags=(
        "--free_gpu_memory_fraction",
        "--kv_cache_free_gpu_memory_fraction",
    ),
    memory_fraction_scope=MEMORY_FRACTION_FREE_AFTER_LOAD,
    default_port=8000,
    health_path="/health",
    metrics_path="/prometheus/metrics",
    image_key="trtllm",
    aliases=("tensorrt", "tensorrt-llm", "tensorrt_llm", "tensorrtllm"),
)

SIM = EngineSpec(
    name="sim",
    launchers=(("llm-d-inference-sim",), ("vllm-sim",)),
    model_flags=("--model",),
    positional_model=False,
    served_model_flags=("--served-model-name",),
    max_model_len_flags=("--max-model-len",),
    block_size_flags=("--block-size",),
    default_port=8000,
    health_path="/health",
    metrics_path="/metrics",
    image_key="llmdInferenceSim",
    aliases=("inference-sim", "llm-d-inference-sim", "vllm-sim"),
)

# Unknown launchers still support common model and port spellings. Their image
# and nonstandard endpoint paths must be stated explicitly.
GENERIC = EngineSpec(
    name="generic",
    launchers=(),
    model_flags=("--model", "--model-path"),
    positional_model=False,
    default_port=8000,
    image_key="vllm",
    aliases=("custom", "other"),
)

#: Registry, in detection priority order. vLLM first because ``vllm serve`` is
#: the most specific two-token signature; GENERIC is never auto-detected.
ENGINE_SPECS: tuple[EngineSpec, ...] = (VLLM, SGLANG, TRTLLM, SIM)

_BY_NAME: dict[str, EngineSpec] = {}
for _spec in (*ENGINE_SPECS, GENERIC):
    _BY_NAME[_spec.name] = _spec
    for _alias in _spec.aliases:
        _BY_NAME[_alias] = _spec


def known_engines() -> list[str]:
    """Canonical engine names, for error messages and CLI help."""
    return [spec.name for spec in (*ENGINE_SPECS, GENERIC)]


def get_engine_spec(name: str | None) -> EngineSpec:
    """Look up an engine by name or alias.

    Unknown names resolve to :data:`GENERIC` rather than raising: an
    unrecognised engine must still be able to stand up, and the caller
    (``resolve_engines``) turns the miss into a warning.
    """
    if not name:
        return VLLM
    return _BY_NAME.get(str(name).strip().lower(), GENERIC)


def is_known_engine(name: str | None) -> bool:
    """True when ``name`` names an engine (or alias) we have a spec for."""
    return bool(name) and str(name).strip().lower() in _BY_NAME


def detect_engine(tokens: Iterable[str]) -> EngineSpec | None:
    """Identify the engine that ``tokens`` launches, or None.

    Matching is by launcher signature, so it works on the user's verbatim
    command with no declaration in the scenario.
    """
    toks = [t for t in tokens if t]
    if not toks:
        return None
    for spec in ENGINE_SPECS:
        if _launcher_index(toks, spec) is not None:
            return spec
    return None


def _launcher_index(tokens: list[str], spec: EngineSpec) -> int | None:
    """Index just past the launcher signature of ``spec`` in ``tokens``."""
    for sig in spec.launchers:
        if len(sig) == 1:
            needle = sig[0]
            for i, tok in enumerate(tokens):
                if tok == needle or tok.rsplit("/", 1)[-1] == needle:
                    return i + 1
        else:
            span = len(sig)
            for i in range(len(tokens) - span + 1):
                window = tokens[i : i + span]
                if (
                    window[0].rsplit("/", 1)[-1] == sig[0]
                    and tuple(window[1:]) == sig[1:]
                ):
                    return i + span
    return None


def launcher_end(tokens: list[str], spec: EngineSpec) -> int | None:
    """Public wrapper over :func:`_launcher_index` (index past the launcher)."""
    return _launcher_index(tokens, spec)
