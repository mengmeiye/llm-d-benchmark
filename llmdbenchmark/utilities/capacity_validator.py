"""Capacity planner validation for a served model against GPU and model constraints.

The inputs are engine-neutral: the context length and the memory fraction come
from whichever flag the role's command spells them in, read by
:mod:`llmdbenchmark.engine.resolver`; the parallelism widths and the device
count come from what the llm-d chart and the pod spec were given. Numbers nobody
stated are ``None``, and the checks that need them are skipped rather than
guessed.

One input is *not* engine-neutral, however much it looks it. Every engine takes
a memory fraction, and every engine means something different by it:

  * vLLM's ``--gpu-memory-utilization`` is the whole device budget -- weights,
    activations and KV together.
  * SGLang's ``--mem-fraction-static`` covers weights and the KV pool only;
    activations and CUDA graphs are taken on top, out of what it left.
  * TRT-LLM's ``--kv_cache_free_gpu_memory_fraction`` is a fraction of what is
    still *free* after weights and peak activation, all of it KV.

So ``0.88`` describes three different KV pools, and reading all three as vLLM's
is not a rounding error: on one 80 GiB device it declares a working SGLang
deployment of Qwen3-32B dead ("cannot serve any requests", because it subtracts
activations a second time), and passes a TRT-LLM one it has understated by 3x.
:attr:`~llmdbenchmark.engine.spec.EngineSpec.memory_fraction_scope` records which
reading an engine takes and :func:`_memory_budget` does that engine's
subtraction; everything downstream -- the verdict, the concurrency estimate, the
suggestions -- follows from it. ``planner.capacity_planner`` supplies the model
and hardware estimates all three share.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

# The leaf engine module, not the `llmdbenchmark.engine` package: the package
# pulls in the command parser and the resolver, and a capacity check has no
# business importing either.
from llmdbenchmark.engine.spec import (
    MEMORY_FRACTION_FREE_AFTER_LOAD,
    MEMORY_FRACTION_WEIGHTS_AND_KV,
    get_engine_spec,
)
from llmdbenchmark.parser.cluster_resource_resolver import (
    effective_accelerator_count,
)
from planner.capacity_planner import (
    KVCacheDetail,
    estimate_vllm_activation_memory,
    estimate_vllm_cuda_graph_memory,
    estimate_vllm_non_torch_memory,
    find_possible_tp,
    get_model_config_from_hf,
    get_text_config,
    gpus_required,
    max_context_len,
    model_memory_req,
    model_total_params,
)

if TYPE_CHECKING:
    from transformers import AutoConfig


class _Logger(Protocol):
    """Minimal logger interface compatible with both project and stdlib loggers."""

    def log_info(self, msg: str, **kwargs: Any) -> None: ...

    def log_warning(self, msg: str, **kwargs: Any) -> None: ...


class _StdlibLoggerAdapter:
    """Wrap a standard :class:`logging.Logger` to match :class:`_Logger`."""

    def __init__(self, logger: logging.Logger) -> None:
        self._log = logger

    def log_info(self, msg: str, **_kw: Any) -> None:
        self._log.info(msg)

    def log_warning(self, msg: str, **_kw: Any) -> None:
        self._log.warning(msg)


def _ensure_logger(logger: Any) -> _Logger:
    """Return a _Logger-compatible object, wrapping stdlib loggers if needed."""
    if hasattr(logger, "log_info") and hasattr(logger, "log_warning"):
        return logger  # type: ignore[return-value]
    return _StdlibLoggerAdapter(logger)


@dataclass
class ValidationParams:
    """Parameters for a single deployment method's capacity validation."""

    models: list[str]
    hf_token: str | None
    replicas: int
    gpu_memory: int  # GPU memory in GB; 0 = unknown (skip GPU memory checks)
    tp: int
    pp: int
    dp: int
    accelerator_nr: int  # User-requested GPUs per pod
    # Both come out of the engine command, which need not state either.
    # 0 = unstated (the engine will use its own default); the checks that
    # need the number are skipped rather than run against a guess.
    gpu_memory_util: float
    max_model_len: int
    # The engine the role's command launches, as `resolve_engines` detected it.
    # Its spec says what `gpu_memory_util` above is a fraction *of* -- the one
    # input whose meaning is engine-specific (see the module docstring). Empty
    # means nobody said, and vLLM's reading is used, which is what this check
    # assumed for every engine before the scope was recorded.
    engine: str = ""
    ignore_failures: bool = False
    label: str = ""  # e.g. "standalone", "decode", "prefill"


def _get_model_config(
    model_name: str,
    hf_token: str | None,
    logger: _Logger,
    ignore_failures: bool = False,
) -> "AutoConfig | None":
    """Fetch model config from HuggingFace, with error handling."""
    tag = "WARNING" if ignore_failures else "ERROR"
    try:
        return get_model_config_from_hf(model_name, hf_token)
    except Exception as exc:
        logger.log_warning(
            f"{tag}: Cannot retrieve model config for {model_name}: {exc}"
        )
        return None


def _convert_accelerator_memory(gpu_name: str, raw_value: str) -> int:
    """Determine GPU memory in GB from an explicit value or GPU product name. Returns 0 if unknown."""
    try:
        return int(raw_value)
    except (ValueError, TypeError):
        pass

    if not gpu_name:
        return 0

    match = re.search(r"(\d+)\s*GB", gpu_name, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match2 = re.search(r"-(\d+)\b", gpu_name)
    if match2:
        return int(match2.group(1))

    return 0


@dataclass(frozen=True)
class _MemoryBudget:
    """Where one replica's GPU memory goes, in GiB, under one engine's reading.

    Every figure covers the whole group of devices a replica occupies
    (``TP x PP x DP``), which is how the planner totals them and how a replica
    actually spends them.
    """

    scope: str
    fraction: float
    total: float  # devices x memory per device
    claimed: float  # what the fraction lets the engine allocate
    weights: float  # model weights, one copy per DP rank
    intermediates: float  # peak activation + CUDA graphs + non-torch overhead
    kv: float  # left for the KV cache; <= 0 means the model cannot serve
    outside: float  # memory the fraction does not claim at all
    activation_per_gpu: float
    non_torch_per_gpu: float
    note: str  # the subtraction, spelled out for the log

    @property
    def intermediates_fit_outside(self) -> bool:
        """Whether what the fraction left can hold what it does not cover.

        Only meaningful for :data:`MEMORY_FRACTION_WEIGHTS_AND_KV`, where
        activations and CUDA graphs are allocated *outside* the fraction: too
        high a value there does not shrink the KV pool, it OOMs mid-forward.
        """
        return self.outside >= self.intermediates


def _memory_budget(
    params: ValidationParams,
    model: str,
    model_config: "AutoConfig",
    spec: Any,
) -> _MemoryBudget:
    """Split a replica's device memory the way ``spec``'s engine splits it.

    The model and hardware estimates are the planner's, unchanged and shared by
    every engine -- weights come from the safetensors index, peak activation and
    overhead from its empirical profiles. What differs is only the subtraction,
    because the fraction the user wrote is a fraction of a different thing in
    each engine (see the module docstring).

    For :data:`MEMORY_FRACTION_DEVICE` the result reproduces
    ``planner.allocatable_kv_cache_memory`` exactly -- deliberately, so vLLM's
    verdict is the one it always was -- except that this does not clamp at zero:
    a model that does not fit reports how far it overran instead of reporting a
    KV pool of 0 GB, which reads as "loads but cannot serve".
    """
    gpu_count = gpus_required(tp=params.tp, pp=params.pp, dp=params.dp)
    total = float(params.gpu_memory) * gpu_count
    fraction = params.gpu_memory_util

    weights = model_memory_req(model, model_config, params.hf_token) * params.dp

    # Scaled as the planner scales them: activation is per replica (one copy per
    # DP rank), the two overheads are per device.
    activation_per_gpu = estimate_vllm_activation_memory(model_config, tp=params.tp)
    non_torch_per_gpu = estimate_vllm_non_torch_memory(params.tp)
    intermediates = (
        activation_per_gpu * params.dp
        + estimate_vllm_cuda_graph_memory() * gpu_count
        + non_torch_per_gpu * gpu_count
    )

    scope = spec.memory_fraction_scope
    flag = (
        spec.memory_util_flags[0] if spec.memory_util_flags else "the memory fraction"
    )

    if scope == MEMORY_FRACTION_WEIGHTS_AND_KV:
        claimed = total * fraction
        kv = claimed - weights
        note = (
            f"{flag}={fraction} covers weights + the KV pool: "
            f"{total:.1f} x {fraction} = {claimed:.2f} GB static, minus "
            f"{weights:.2f} GB weights = {kv:.2f} GB for KV. Activations and "
            f"overhead ({intermediates:.2f} GB) are taken on top, out of the "
            f"{total - claimed:.2f} GB the fraction leaves."
        )
    elif scope == MEMORY_FRACTION_FREE_AFTER_LOAD:
        free = total - weights - intermediates
        kv = free * fraction
        claimed = weights + intermediates + kv
        note = (
            f"{flag}={fraction} is a fraction of what is free once the engine "
            f"has loaded: {total:.1f} minus {weights:.2f} GB weights and "
            f"{intermediates:.2f} GB activations/overhead = {free:.2f} GB free, "
            f"x {fraction} = {kv:.2f} GB for KV."
        )
    else:
        claimed = total * fraction
        kv = claimed - weights - intermediates
        note = (
            f"{flag}={fraction} is the whole device budget: {total:.1f} x "
            f"{fraction} = {claimed:.2f} GB, minus {weights:.2f} GB weights and "
            f"{intermediates:.2f} GB activations/overhead = {kv:.2f} GB for KV."
        )

    return _MemoryBudget(
        scope=scope,
        fraction=fraction,
        total=total,
        claimed=claimed,
        weights=weights,
        intermediates=intermediates,
        kv=kv,
        outside=total - claimed,
        activation_per_gpu=activation_per_gpu,
        non_torch_per_gpu=non_torch_per_gpu,
        note=note,
    )


def validate_vllm_params(
    params: ValidationParams,
    logger: _Logger,
) -> list[str]:
    """Validate a role's engine parameters against the capacity planner."""
    tag = "WARNING" if params.ignore_failures else "ERROR"
    prefix = f"[{params.label}] " if params.label else ""
    messages: list[str] = []

    # What the role's engine means by the memory fraction, and what to call that
    # flag in a message. An engine with no spec resolves to GENERIC, which states
    # no fraction flag: `gpu_memory_util` is then 0 and every check below that
    # reads it turns itself off anyway.
    spec = get_engine_spec(params.engine)
    fraction_flag = (
        spec.memory_util_flags[0] if spec.memory_util_flags else "the memory fraction"
    )

    def msg(text: str) -> None:
        full = f"{prefix}{tag}: {text}"
        messages.append(full)
        logger.log_warning(full)

    def info(text: str) -> None:
        full = f"{prefix}{text}"
        messages.append(full)
        logger.log_info(full)

    per_replica_gpus = gpus_required(tp=params.tp, pp=params.pp, dp=params.dp)
    if params.replicas == 0:
        per_replica_gpus = 0

    if per_replica_gpus > params.accelerator_nr:
        msg(
            f"Accelerator requested is {params.accelerator_nr} but "
            f"TP x PP x DP = {params.tp} x {params.pp} x {params.dp} "
            f"= {per_replica_gpus} GPUs are required per replica"
        )

    if 0 < per_replica_gpus < params.accelerator_nr:
        msg(
            f"Each replica requires {per_replica_gpus} GPUs, but "
            f"{params.accelerator_nr} requested per pod. "
            f"Some GPUs will be idle."
        )

    skip_gpu_tests = False
    if params.gpu_memory is None or params.gpu_memory == 0:
        info(
            "Cannot determine accelerator memory. "
            "Set accelerator.memory in your config to enable "
            "GPU memory validation (KV cache estimation). "
            "Skipping GPU memory checks."
        )
        skip_gpu_tests = True

    if not params.gpu_memory_util:
        info(
            f"The {spec.name} command does not set a GPU memory fraction "
            f"({fraction_flag}), so how much of each device the engine will "
            "claim is unknown. Skipping KV-cache estimation."
        )
        skip_gpu_tests = True

    if not params.max_model_len:
        info(
            "The engine command does not set a context length "
            "(vLLM --max-model-len, SGLang --context-length, TRT-LLM "
            "--max_seq_len), so the engine will use the model's own maximum. "
            "Skipping context-length and KV-cache checks."
        )
        skip_gpu_tests = True

    for model in params.models:
        model_config = _get_model_config(
            model, params.hf_token, logger, params.ignore_failures
        )
        text_config = None
        if model_config is not None:
            text_config = get_text_config(model_config)

        if model_config is not None:
            try:
                valid_tp = find_possible_tp(text_config)
                if params.tp not in valid_tp:
                    msg(
                        f"TP={params.tp} is invalid for {model}. "
                        f"Valid values: {valid_tp}"
                    )
            except AttributeError:
                msg(
                    f"Cannot determine valid TP values for {model} "
                    "(num_attention_heads not available)"
                )

            valid_max_ctx = 0
            try:
                valid_max_ctx = max_context_len(model_config)
            except AttributeError as exc:
                msg(f"Cannot determine max context length for {model}: {exc}")

            if (
                valid_max_ctx
                and params.max_model_len
                and params.max_model_len > valid_max_ctx
            ):
                msg(
                    f"maxModelLen={params.max_model_len} exceeds "
                    f"model limit of {valid_max_ctx} for {model}"
                )
        else:
            msg("Model config on parameter shape not available.")

        if not skip_gpu_tests:
            # What the fraction *claims* is stated with the budget below, once
            # the model's weights are known: how much of the claim is left for
            # KV depends on which of the three readings this engine takes, so
            # there is no engine-neutral "available memory" to report here.
            info(
                f"{params.gpu_memory} GB per GPU x {per_replica_gpus} GPU(s) "
                f"per replica = {params.gpu_memory * per_replica_gpus} GB, "
                f"of which {spec.name} is told to take "
                f"{fraction_flag}={params.gpu_memory_util}"
            )

        if model_config is not None:
            try:
                total_params = model_total_params(model, params.hf_token)
                info(f"{model} has {total_params:,} parameters")

                model_mem = model_memory_req(model, model_config, params.hf_token)
                info(f"{model} requires {model_mem:.2f} GB of memory")

                if not skip_gpu_tests:
                    budget = _memory_budget(params, model, model_config, spec)
                    info(
                        f"Peak activation memory per GPU: "
                        f"{budget.activation_per_gpu:.2f} GB"
                    )
                    info(f"Non-torch memory per GPU: {budget.non_torch_per_gpu:.2f} GB")
                    info(budget.note)

                    avail_kv = budget.kv
                    kv_details = KVCacheDetail(
                        model, model_config, params.max_model_len, batch_size=1
                    )
                    per_req_kv = kv_details.per_request_kv_cache_gb

                    # Only this engine's own reading puts the intermediates
                    # outside the fraction, and only there can a value be too
                    # *high* without shrinking the KV pool: the pool is sized
                    # fine and the forward pass has nowhere to run. It is what
                    # OOMs SGLang at 0.95 on an 80 GiB device.
                    if (
                        budget.scope == MEMORY_FRACTION_WEIGHTS_AND_KV
                        and not budget.intermediates_fit_outside
                    ):
                        msg("DEPLOYMENT WILL FAIL: no room left for activations.")
                        msg(
                            f"{fraction_flag}={budget.fraction} reserves "
                            f"{budget.claimed:.2f} GB of {budget.total:.1f} GB "
                            f"for weights and KV, leaving {budget.outside:.2f} "
                            f"GB -- but activations, CUDA graphs and allocator "
                            f"overhead need {budget.intermediates:.2f} GB on "
                            f"top of it, and {spec.name} allocates those "
                            f"outside the fraction."
                        )
                        _log_config_suggestions(
                            msg,
                            params,
                            f"4. Reduce {fraction_flag} to at most "
                            f"{max(0.0, (budget.total - budget.intermediates) / budget.total):.2f} "
                            f"(currently {budget.fraction}), which is what "
                            f"leaves room for them",
                        )

                    elif avail_kv <= 0:
                        msg(
                            "DEPLOYMENT WILL FAIL: Insufficient GPU memory "
                            "to load model."
                        )
                        msg(
                            f"Model requires {abs(avail_kv):.2f} GB MORE "
                            "memory than available after loading weights "
                            "and activation memory."
                        )
                        _log_config_suggestions(msg, params, _raise_fraction(budget))

                    elif avail_kv < per_req_kv:
                        msg(
                            "DEPLOYMENT WILL FAIL: Model loads but cannot "
                            "serve any requests."
                        )
                        msg(
                            f"Available KV cache: {avail_kv:.2f} GB, "
                            f"required per request "
                            f"(max_model_len={params.max_model_len}): "
                            f"{per_req_kv:.2f} GB"
                        )
                        _log_config_suggestions(msg, params, _raise_fraction(budget))

                    else:
                        info(f"Allocatable KV cache memory: {avail_kv:.2f} GB")
                        info(
                            f"Per-request KV cache "
                            f"(max_model_len={params.max_model_len}): "
                            f"{per_req_kv:.2f} GB"
                        )

                        # floor(pool / per request), which is what
                        # `planner.max_concurrent_requests` computes -- done here
                        # so it divides *this* engine's pool rather than
                        # recomputing a vLLM-shaped one.
                        total_concurrent = (
                            math.floor(avail_kv / per_req_kv) if per_req_kv else 0
                        )
                        info(
                            f"Max concurrent requests (worst case, "
                            f"each at max_model_len): {total_concurrent}"
                        )

            except (AttributeError, Exception) as exc:
                msg(f"Cannot estimate model memory or KV cache for {model}: {exc}")
        else:
            msg("Model architecture info not available -- skipping memory checks.")

    return messages


def _raise_fraction(budget: _MemoryBudget) -> str:
    """The "give the engine more memory" suggestion, in this engine's terms."""
    if budget.scope == MEMORY_FRACTION_FREE_AFTER_LOAD:
        # This fraction is of free memory, so raising it takes from a reserve the
        # engine deliberately left; 1.0 means "all of it".
        return (
            f"4. Increase the memory fraction (currently {budget.fraction}) "
            f"toward 1.0, which leaves no headroom outside the KV pool"
        )
    return (
        f"4. Increase the memory fraction (currently {budget.fraction}, may cause OOM)"
    )


def _log_config_suggestions(
    msg_fn, params: ValidationParams, fraction_advice: str | None = None
) -> None:
    """Log configuration suggestions when deployment will fail.

    ``fraction_advice`` is the last line, which is the only engine-specific one:
    whether to raise or lower the memory fraction, and toward what, depends on
    what that fraction measures.
    """
    spec = get_engine_spec(params.engine)
    flag = spec.memory_util_flags[0] if spec.memory_util_flags else "memory fraction"

    msg_fn("  Current config:")
    msg_fn(f"    engine: {spec.name}")
    msg_fn(f"    GPU memory per device: {params.gpu_memory} GB")
    msg_fn(f"    {flag}: {params.gpu_memory_util}")
    msg_fn(f"    maxModelLen: {params.max_model_len}")
    msg_fn(f"    TP: {params.tp}, PP: {params.pp}, DP: {params.dp}")
    msg_fn("  Possible solutions:")
    msg_fn(f"    1. Reduce maxModelLen (currently {params.max_model_len})")
    msg_fn("    2. Increase tensor parallelism to use more GPUs")
    msg_fn("    3. Use GPUs with more memory")
    msg_fn(
        "    "
        + (
            fraction_advice
            or f"4. Increase the memory fraction "
            f"(currently {params.gpu_memory_util}, may cause OOM)"
        )
    )


def _extract_params(
    plan_config: dict,
    method: str,
    ignore_failures: bool,
) -> ValidationParams | None:
    """Extract ValidationParams for a deployment method from the plan config."""
    method_config = plan_config.get(method, {})

    replicas = int(method_config.get("replicas", 0))
    if replicas == 0:
        return None
    if method_config.get("enabled") is False:
        return None

    model_config = plan_config.get("model", {})
    model_name = model_config.get("huggingfaceId") or model_config.get("name", "")
    if not model_name:
        return None
    models = [m.strip() for m in model_name.split(",") if m.strip()]

    hf_section = plan_config.get("huggingface", {})
    hf_token = hf_section.get("token") or os.environ.get("HF_TOKEN") or None
    if hf_token in ("", "REPLACE_TOKEN"):
        hf_token = None

    # KV cache is sharded across the widths the llm-d chart was given, which is
    # where a scenario states them (the chart needs them for LeaderWorkerSet
    # sizing and wide-EP layout). Unstated means 1.
    parallelism = method_config.get("parallelism") or {}

    def width(key: str) -> int:
        value = parallelism.get(key)
        return int(value) if value else 1

    tp = width("tensor")
    pp = width("pipeline")
    dp = width("dataLocal") or width("data")

    accel_section = plan_config.get("accelerator", {})

    # How many devices one pod actually holds, from the Kubernetes request the
    # role writes down (same chain the manifests render).
    stated_nr, _ = effective_accelerator_count(method_config, plan_config)
    accelerator_nr = int(method_config.get("acceleratorNr", stated_nr or tp * pp * dp))
    accel_type = method_config.get("acceleratorType", {}).get(
        "labelValue", ""
    ) or accel_section.get("type", "")
    gpu_memory = _convert_accelerator_memory(
        accel_type,
        str(accel_section.get("memory", "")),
    )

    # Both are read out of the engine command; a command that does not set them
    # leaves them None (the engine then uses its own default, which we cannot
    # know). Zero is the "unknown" sentinel the checks below already understand:
    # it turns off GPU-memory and context-length validation instead of
    # validating against a number nobody asked for.
    try:
        gpu_memory_util = float(model_config.get("gpuMemoryUtilization") or 0.0)
    except (TypeError, ValueError):
        gpu_memory_util = 0.0
    try:
        max_model_len = int(model_config.get("maxModelLen") or 0)
    except (TypeError, ValueError):
        max_model_len = 0

    # Which engine this role launches, as `resolve_engines` detected it from the
    # command (`engine.name`, normalised: an alias is already resolved). It says
    # what `gpu_memory_util` above is a fraction of. Absent -- an older plan, or
    # a role with no engine block -- leaves it empty and vLLM's reading stands.
    role_engine = method_config.get("engine") or {}
    engine = role_engine.get("name", "") if isinstance(role_engine, dict) else ""

    return ValidationParams(
        models=models,
        hf_token=hf_token,
        replicas=replicas,
        gpu_memory=gpu_memory,
        tp=tp,
        pp=pp,
        dp=dp,
        accelerator_nr=accelerator_nr,
        gpu_memory_util=gpu_memory_util,
        max_model_len=max_model_len,
        engine=str(engine or ""),
        ignore_failures=ignore_failures,
        label=method,
    )


def run_capacity_planner(
    plan_config: dict,
    logger: Any,
    ignore_failures: bool = False,
) -> list[str]:
    """Run capacity planner validation for all active deployment methods."""
    log = _ensure_logger(logger)
    all_messages: list[str] = []

    if ignore_failures:
        log.log_info(
            "Validating engine configuration against Capacity Planner "
            "(deployment will continue even if validation fails)"
        )
    else:
        log.log_info(
            "Validating engine configuration against Capacity Planner "
            "(deployment will halt if validation fails)"
        )

    is_fma = plan_config.get("fma", {}).get("enabled", False)
    if is_fma:
        log.log_info("Deployment method is fma -- skipping engine capacity validation")
        return all_messages

    standalone = plan_config.get("standalone", {})
    is_standalone = (
        standalone.get("enabled", False) and int(standalone.get("replicas", 0)) > 0
    )

    if is_standalone:
        log.log_info("Deployment method is standalone")
        params = _extract_params(plan_config, "standalone", ignore_failures)
        if params:
            all_messages.extend(validate_vllm_params(params, log))
    else:
        log.log_info(
            "Deployment method is modelservice -- "
            "checking decode and prefill configurations"
        )

        for method in ("decode", "prefill"):
            params = _extract_params(plan_config, method, ignore_failures)
            if params:
                log.log_info(
                    f"Validating {method} engine arguments for {params.models} ..."
                )
                all_messages.extend(validate_vllm_params(params, log))
            else:
                log.log_info(f"{method} is disabled or has 0 replicas -- skipping")

    return all_messages
