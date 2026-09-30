"""Inference-engine registry.

An engine's command line is the user's to write, verbatim (see
:mod:`llmdbenchmark.engine.command`). This module holds the only engine
knowledge llm-d-benchmark cannot do without, which is small on purpose:

* how to recognise the server invocation inside a shell snippet, so a preamble
  (``export ...; source ...;``) can be told apart from the launch;
* which flag carries the model reference and the bind port -- the two facts
  Kubernetes needs *before* the process exists, because the Service and the
  probes have to name a port and the model volume has to name a model;
* which flag carries the context length and the memory fraction, the two
  numbers the pre-deploy capacity check reads;
* engine-neutral defaults (health path, metrics path, default port, image key)
  used when the command says nothing.

Everything else -- parallelism, quantization, scheduler knobs, cache policy,
attention backend -- is the engine's business and passes through untouched.
Adding an engine means adding one :class:`EngineSpec` here: no new template
branch, no new scenario key, no new flag table.

Flag names come from each engine's own argument definitions:

* vLLM                -- ``vllm/entrypoints/openai/cli_args.py`` + ``EngineArgs``
* SGLang              -- ``python/sglang/srt/arg_groups/fields/*.py``
* TensorRT-LLM        -- ``tensorrt_llm/commands/serve.py`` (underscore flags)
* llm-d-inference-sim -- ``--model`` / ``--port``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

#: Values for :attr:`EngineSpec.memoryFractionScope` -- what an engine's memory
#: fraction is a fraction *of*. One number, three denominators:
#:
#: ``DEVICE``
#:     The whole device budget: weights, activations and the KV cache together.
#:     vLLM's ``--gpu-memory-utilization`` ("the fraction of GPU memory to be
#:     used for the model executor").
#: ``WEIGHTS_AND_KV``
#:     Weights plus the KV pool only. Activations, CUDA graphs and allocator
#:     overhead are taken *on top*, out of what the fraction left behind.
#:     SGLang's ``--mem-fraction-static`` ("the fraction of the memory used for
#:     static allocation (model weights and KV cache memory pool)").
#: ``FREE_AFTER_LOAD``
#:     The fraction of whatever is still free once weights and peak activation
#:     are in place, all of it KV. TRT-LLM's
#:     ``--kv_cache_free_gpu_memory_fraction`` ("the fraction of free GPU memory
#:     to be used for the KV cache").
#:
#: The same 0.88 therefore means three different KV pools, which is why the
#: pre-deploy capacity check reads this rather than assuming vLLM's reading --
#: see :mod:`llmdbenchmark.utilities.capacity_validator`.
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

    # ---- recognising the launch inside a shell snippet ---------------------
    # Each entry is a token sequence that introduces the server. A snippet
    # matches when its tokens contain the sequence in order and adjacent
    # (``vllm serve``), or -- for single-token entries -- when a token's
    # basename equals it (``trtllm-serve``, ``/usr/bin/trtllm-serve``).
    launchers: tuple[tuple[str, ...], ...] = ()

    # Subcommands of a launcher that is a command group. ``trtllm-serve serve
    # <model>`` and ``trtllm-serve <model>`` are the same invocation -- both
    # appear in the llm-d guides -- so the subcommand is skipped before the
    # positionals are read, or it would be taken for the model.
    subcommands: tuple[str, ...] = ()

    # ---- the model the command serves -------------------------------------
    # ``positionalModel`` means the engine also accepts the model as a bare
    # positional right after the launcher (vLLM, TRT-LLM).
    modelFlags: tuple[str, ...] = ()
    positionalModel: bool = False
    servedModelFlags: tuple[str, ...] = ()

    # ---- the port the engine binds ----------------------------------------
    # Every engine here spells this ``--port``; the tuple exists so one that
    # does not can be added without touching the parser.
    portFlags: tuple[str, ...] = ("--port",)

    # ---- numbers read back onto ``model.*`` -------------------------------
    # KV-cache headroom needs the context window and the fraction of device
    # memory the engine is allowed to take; the router's prefix-cache index
    # needs the KV page size, because it rebuilds block hashes on the same
    # boundaries the engine writes. All three are read from the flag the user
    # already wrote, so the scenario does not restate them; a command that omits
    # one leaves it unknown and whatever needed it turns itself off.
    maxModelLenFlags: tuple[str, ...] = ()
    memoryUtilFlags: tuple[str, ...] = ()
    blockSizeFlags: tuple[str, ...] = ()

    # What the fraction above measures. Engines spell one number three ways and
    # mean three different splits of the device by it, so the capacity check
    # cannot subtract the same things from all of them: see the scope constants
    # at the top of this module. An engine with no `memoryUtilFlags` never has
    # this read -- nothing states a fraction to interpret.
    memoryFractionScope: str = MEMORY_FRACTION_DEVICE

    # ---- engine-neutral defaults ------------------------------------------
    defaultPort: int = 8000
    healthPath: str = "/health"
    metricsPath: str = "/metrics"

    # Key under ``images:`` in defaults.yaml supplying the default image.
    imageKey: str = "vllm"

    # Human-facing aliases accepted for ``engine.name`` in a scenario.
    aliases: tuple[str, ...] = ()

    def flags_for_metric(self, metric: str) -> tuple[str, ...]:
        """Return the flag spellings for one ``MODEL_READS`` entry."""
        return {
            "maxModelLen": self.maxModelLenFlags,
            "gpuMemoryUtilization": self.memoryUtilFlags,
            "blockSize": self.blockSizeFlags,
        }.get(metric, ())


VLLM = EngineSpec(
    name="vllm",
    launchers=(
        ("vllm", "serve"),
        ("vllm", "bench"),
        ("-m", "vllm.entrypoints.openai.api_server"),
    ),
    modelFlags=("--model",),
    positionalModel=True,
    servedModelFlags=("--served-model-name",),
    maxModelLenFlags=("--max-model-len",),
    memoryUtilFlags=("--gpu-memory-utilization",),
    memoryFractionScope=MEMORY_FRACTION_DEVICE,
    blockSizeFlags=("--block-size",),
    defaultPort=8000,
    healthPath="/health",
    metricsPath="/metrics",
    imageKey="vllm",
)

SGLANG = EngineSpec(
    name="sglang",
    launchers=(
        ("-m", "sglang.launch_server"),
        ("sglang.launch_server",),
    ),
    modelFlags=("--model-path", "--model"),
    positionalModel=False,
    servedModelFlags=("--served-model-name",),
    maxModelLenFlags=("--context-length",),
    memoryUtilFlags=("--mem-fraction-static",),
    # Weights and the KV pool; activations and CUDA graphs come out of the rest
    # of the device, which is why a fraction vLLM is happy with can OOM SGLang
    # mid-forward.
    memoryFractionScope=MEMORY_FRACTION_WEIGHTS_AND_KV,
    # SGLang calls a KV block a page ("The number of tokens in a page").
    blockSizeFlags=("--page-size",),
    defaultPort=30000,
    healthPath="/health",
    metricsPath="/metrics",
    imageKey="sglang",
)

TRTLLM = EngineSpec(
    name="trtllm",
    launchers=(("trtllm-serve",),),
    # trtllm-serve is a click group whose unrecognised first argument falls
    # through to `serve` (see DefaultGroup in tensorrt_llm/commands/serve.py),
    # so the llm-d guides spell it `trtllm-serve serve <model>`.
    subcommands=(
        "serve",
        "disaggregated",
        "disaggregated_mpi_worker",
        "mm_embedding_serve",
        "embeddings",
    ),
    # trtllm-serve takes the model as a click argument; --model_path exists on
    # some subcommands. Underscore spelling is TRT-LLM's convention.
    modelFlags=("--model", "--model_path"),
    positionalModel=True,
    servedModelFlags=("--served_model_name", "--served-model-name"),
    maxModelLenFlags=("--max_seq_len",),
    # trtllm-serve declares one option under two names
    # (`@stability_option("--free_gpu_memory_fraction",
    # "--kv_cache_free_gpu_memory_fraction", ...)`), and the llm-d
    # optimized-baseline TRT-LLM guide writes the longer one. Both are read.
    memoryUtilFlags=(
        "--free_gpu_memory_fraction",
        "--kv_cache_free_gpu_memory_fraction",
    ),
    # "free" is what is left after the engine has loaded and profiled, so this
    # fraction is taken of a much smaller number than vLLM's or SGLang's -- and
    # 0.9 of it is a normal, safe setting rather than an aggressive one.
    memoryFractionScope=MEMORY_FRACTION_FREE_AFTER_LOAD,
    defaultPort=8000,
    healthPath="/health",
    metricsPath="/prometheus/metrics",
    imageKey="trtllm",
    aliases=("tensorrt", "tensorrt-llm", "tensorrt_llm", "tensorrtllm"),
)

SIM = EngineSpec(
    name="sim",
    launchers=(("llm-d-inference-sim",), ("vllm-sim",)),
    modelFlags=("--model",),
    positionalModel=False,
    servedModelFlags=("--served-model-name",),
    maxModelLenFlags=("--max-model-len",),
    blockSizeFlags=("--block-size",),
    defaultPort=8000,
    healthPath="/health",
    metricsPath="/metrics",
    imageKey="llmdInferenceSim",
    aliases=("inference-sim", "llm-d-inference-sim", "vllm-sim"),
)

# Escape hatch: an engine we have no spec for. The command still runs verbatim;
# only the derived facts are unavailable, so the scenario states `engine.port`
# (or the command writes `--port $ENGINE_PORT`) instead.
GENERIC = EngineSpec(
    name="generic",
    launchers=(),
    modelFlags=("--model", "--model-path"),
    positionalModel=False,
    defaultPort=8000,
    imageKey="vllm",
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
                if tuple(tokens[i : i + span]) == sig:
                    return i + span
    return None


def launcher_end(tokens: list[str], spec: EngineSpec) -> int | None:
    """Public wrapper over :func:`_launcher_index` (index past the launcher)."""
    return _launcher_index(tokens, spec)
