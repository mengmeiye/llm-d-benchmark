"""Pydantic v2 config validation for the LLM-D benchmark rendering pipeline.

Validates the merged config dict (defaults + scenario) after resolvers run,
before Jinja template rendering.  Validation is non-blocking: errors are
collected and returned as warning strings, never raised as exceptions.

Phase 1 covers the most commonly overridden and error-prone sections:
model, decode, prefill, engine, harness, and top-level parallelism.

The root model uses ``extra="allow"`` so that unmodeled top-level keys
pass through without error.  Nested section models use ``extra="forbid"``
to catch typos within modeled sections.

Fields do not carry default values -- ``defaults.yaml`` is the single source
of truth for defaults.  The schema only defines types and constraints.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ---------------------------------------------------------------------------
# Shared ConfigDict presets
# ---------------------------------------------------------------------------

STRICT_CONFIG = ConfigDict(
    extra="forbid",
    validate_assignment=True,
    str_strip_whitespace=True,
)

LENIENT_CONFIG = ConfigDict(
    extra="allow",
    validate_assignment=True,
    str_strip_whitespace=True,
)

# ---------------------------------------------------------------------------
# Shared / reusable sub-models
# ---------------------------------------------------------------------------


class DescriptionConfig(BaseModel):
    """Submitter-provided run label, surfaced as run.description/keywords."""

    model_config = STRICT_CONFIG

    text: str = ""
    keywords: list[str] = Field(default_factory=list)


class ParallelismConfig(BaseModel):
    """Widths the llm-d chart needs to lay a role out across pods and nodes.

    These are chart values, not engine flags: the modelservice chart sizes its
    LeaderWorkerSet and its wide-EP layout from them, and the pre-deploy
    capacity check shards KV cache by them. The engine learns its own widths
    from the command line like every other flag, so a single-pod role leaves
    these at 1 and says nothing.
    """

    model_config = STRICT_CONFIG

    data: int = Field(ge=0)
    dataLocal: int = Field(ge=0)
    tensor: int = Field(ge=0)
    workers: int = Field(ge=0)
    # Not in defaults.yaml and rarely set, so defaulted rather than required.
    pipeline: int = Field(default=1, ge=0)


class ResourceQuantities(BaseModel):
    """Container resource limits/requests.

    Uses ``extra="allow"`` because arbitrary accelerator keys
    (``nvidia.com/gpu``, ``ibm.com/spyre_vf``, etc.) are valid here.
    """

    model_config = LENIENT_CONFIG

    memory: str | int
    cpu: str | int


class ResourcesConfig(BaseModel):
    """Resource configuration (limits + requests)."""

    model_config = STRICT_CONFIG

    limits: ResourceQuantities
    requests: ResourceQuantities


class ProbeConfig(BaseModel):
    """Health probe (startup / liveness / readiness)."""

    model_config = STRICT_CONFIG

    path: str | None = None
    # Optional explicit port (defaults to the role's effective vLLM port at
    # render time). Set per-probe in the scenario when the probe needs to
    # hit a non-default port (e.g. uds-tokenizer health on 8082).
    port: int | str | None = None
    failureThreshold: int = Field(ge=1)
    initialDelaySeconds: int | None = Field(default=None, ge=0)
    periodSeconds: int = Field(ge=1)
    timeoutSeconds: int | None = Field(default=None, ge=1)


class ProbesConfig(BaseModel):
    """Container probe configuration."""

    model_config = STRICT_CONFIG

    startup: ProbeConfig
    liveness: ProbeConfig
    readiness: ProbeConfig


class AcceleratorTypeConfig(BaseModel):
    """Node selector for accelerator type (GPU label matching)."""

    model_config = STRICT_CONFIG

    # The cluster resolver deliberately removes both fields when a device
    # resource is available but no portable SKU label exists. In that case
    # Kubernetes schedules from the accelerator resource request alone.
    labelKey: str | None = None
    labelValue: str | None = None
    labelValues: list[str] | None = None


class AcceleratorConfig(BaseModel):
    """Scenario-level accelerator override (count / resourceName).

    Distinct from ``AcceleratorTypeConfig`` which is for node selection.
    Scenarios use this for resource quantity overrides.
    """

    model_config = LENIENT_CONFIG

    count: int | str | None = None
    resourceName: str | None = None
    memory: str | None = None


class PodMonitorConfig(BaseModel):
    """PodMonitor configuration for Prometheus scraping."""

    model_config = STRICT_CONFIG

    enabled: bool
    portName: str
    # Where the engine publishes Prometheus metrics. Unset by default: the path
    # is engine knowledge, so it comes from `<role>.engine.metricsPath`. Set it
    # only to scrape somewhere else.
    path: str | None = None
    interval: str
    scrapeTimeout: str | None = None
    labels: dict[str, str]
    annotations: dict[str, str]
    relabelings: list[Any]
    metricRelabelings: list[Any]


class DeploymentMonitoringConfig(BaseModel):
    """Monitoring block for decode/prefill deployments."""

    model_config = STRICT_CONFIG

    podmonitor: PodMonitorConfig


class AutoscalingConfig(BaseModel):
    """Horizontal pod autoscaler configuration."""

    model_config = STRICT_CONFIG

    enabled: bool
    minReplicas: int | None = None
    maxReplicas: int | None = None


# ---------------------------------------------------------------------------
# Engine config (inside decode/prefill/standalone/nok8s)
# ---------------------------------------------------------------------------


class EngineImageConfig(BaseModel):
    """Per-role engine image override.

    Whatever a role omits is filled from ``images.<engine>`` by
    :func:`llmdbenchmark.engine.resolver.resolve_engines`.
    """

    model_config = STRICT_CONFIG

    repository: str | None = None
    tag: str | None = None
    pullPolicy: str | None = None


class EngineConfig(BaseModel):
    """The inference engine a role runs.

    Two fields are all a scenario normally writes: ``command`` (the launch
    line, verbatim, exactly as it would be typed at a shell) and -- only when
    there is no command to read -- ``name``. ``extraArgs`` is the third, for a
    stack that wants a shared command plus a couple of words of its own.
    Everything else is either an escape hatch for an image whose entrypoint
    fixes a value, or a fact ``resolve_engines`` read out of the command and
    recorded here for the templates.
    """

    model_config = STRICT_CONFIG

    # -- stated by the scenario --------------------------------------------
    name: str | None = None
    command: str | None = None
    #: Words appended to ``command``, unexamined. For the one shape a verbatim
    #: command cannot serve: several stacks sharing one launch line where a few
    #: differ in a flag or two. Repeating a flag the command already carries
    #: overrides it, because every engine's argument parser keeps the last
    #: occurrence -- and so does the reader in ``llmdbenchmark.engine.command``,
    #: so the numbers read back match what the engine gets. Nothing here knows
    #: what any of the words mean.
    extraArgs: list[str] = Field(default_factory=list)
    #: Overrides the ``--port`` in the command. Needed only when the image's
    #: entrypoint fixes the port, or when the command could not be read.
    port: int | None = None
    #: Runs before the engine in the same container; overrides
    #: ``engine.preprocessScript`` for this role.
    preprocessCommand: str | None = None
    #: ``custom`` (render ``command``) or ``imageDefault`` (run the image's
    #: entrypoint). Defaulted from whether a command is present.
    modelCommand: str | None = None
    #: Args appended to the image entrypoint in ``imageDefault`` mode.
    args: list[str] = Field(default_factory=list)
    image: EngineImageConfig | None = None
    imagePullPolicy: str | None = None
    containerName: str | None = None

    # -- written by resolve_engines (not user-set) --------------------------
    #: Everything read out of the command, as published by
    #: ``ParsedCommand.to_dict()``. Consumed by steps and smoketests so they
    #: never re-parse the command themselves.
    facts: dict[str, Any] | None = None
    #: Where the engine answers health checks and exposes Prometheus metrics.
    #: Probes and the PodMonitor read these instead of naming a vLLM path.
    healthPath: str | None = None
    metricsPath: str | None = None
    imageKey: str | None = None


# ---------------------------------------------------------------------------
# Deployment base (shared by decode/prefill)
# ---------------------------------------------------------------------------


class DeploymentBaseConfig(BaseModel):
    """Shared structure for decode and prefill deployment sections."""

    model_config = STRICT_CONFIG

    enabled: bool
    replicas: int = Field(ge=0)

    autoscaling: AutoscalingConfig
    nodeSelector: dict[str, str]
    schedulerName: str | None = None
    priorityClassName: str | None = None
    ephemeralStorage: str | None = None
    networkResource: str | None = None
    networkNr: str | None = None

    acceleratorType: AcceleratorTypeConfig
    accelerator: AcceleratorConfig | None = None

    parallelism: ParallelismConfig
    resources: ResourcesConfig
    # Pod-level securityContext (e.g. supplementalGroups for /dev/dri access on
    # Intel XPU nodes). Distinct from the container-level securityContext under
    # ``extraContainerConfig`` -- supplementalGroups is a Pod field.
    podSecurityContext: dict[str, Any] | None = None
    shm: dict[str, str] | None = None
    probes: ProbesConfig
    engine: EngineConfig

    mountModelVolume: bool
    additionalVolumeMounts: list[Any]
    additionalVolumes: list[Any]
    extraEnvVars: list[dict[str, Any]]
    extraContainerConfig: dict[str, Any]
    extraPodConfig: dict[str, Any]
    initContainers: list[Any]
    # Container-level ports (e.g. [{containerPort: 8200, name: http}]) when
    # the scenario needs to expose a port the chart wouldn't add by default.
    ports: list[dict[str, Any]] | None = None
    monitoring: DeploymentMonitoringConfig

    # Per-pod context-length labels, one per replica. To vary an engine
    # parameter per replica, write a `,,`-delimited value in `extraEnvVars`
    # and reference the variable from the engine command -- the preprocess
    # splits it by pod index.
    contextLengthRanges: list[str] = Field(default_factory=list)

    annotations: dict[str, str] | None = None
    tolerations: list[dict[str, Any]] | None = None

    hostIPC: bool | None = None
    hostPID: bool | None = None
    enableServiceLinks: bool | None = None
    terminationGracePeriodSeconds: int | None = None
    subGroupPolicy: dict[str, Any] | None = None
    subGroupExclusiveTopology: bool | None = None


class DecodeConfig(DeploymentBaseConfig):
    """Decode-specific configuration."""

    model_config = STRICT_CONFIG


class PrefillConfig(DeploymentBaseConfig):
    """Prefill-specific configuration."""

    model_config = STRICT_CONFIG


# ---------------------------------------------------------------------------
# engine (plan-wide)
# ---------------------------------------------------------------------------


class EngineCommonConfig(BaseModel):
    """Engine settings shared by every role and deployment method.

    This is the pod-shaped half of running an engine -- where HOME points, which
    shell wraps the command, which port the Service exposes, what gets mounted.
    No engine *parameters* live here: those belong in the role's ``command``,
    which llm-d-benchmark renders verbatim. ``name`` and ``command`` are here
    only so a single-engine plan can state them once instead of per role.
    """

    model_config = STRICT_CONFIG

    #: Default engine for roles that do not name one. Usually left null and
    #: detected from the command.
    name: str | None = None
    #: Default launch command for roles that do not carry their own.
    command: str | None = None
    #: Default ``extraArgs`` for roles that do not carry their own. A role's own
    #: list replaces this one rather than adding to it, the same as ``command``.
    extraArgs: list[str] = Field(default_factory=list)
    #: Name of the serving container. Defaults to llm-d's engine-neutral
    #: ``modelserver`` so manifests read the same whichever engine runs.
    containerName: str | None = None
    #: Port the Service exposes (distinct from the port the engine binds,
    #: which comes from the command).
    servicePort: int
    #: Shell used to exec the command (``<shell> -c "<command>"``).
    shell: str
    #: Runs before the engine in every role that does not override it.
    preprocessScript: str

    priorityClassName: str
    pullSecret: str
    containerHome: str
    hfHome: str
    ephemeralStorageResource: str
    ephemeralStorage: str
    networkResource: str
    networkNr: str
    volumes: list[dict[str, Any]]
    volumeMounts: list[dict[str, Any]]

    shmMemory: str | None = None
    podScheduler: str | None = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Model configuration."""

    model_config = STRICT_CONFIG

    name: str
    shortName: str
    path: str
    huggingfaceId: str
    size: str

    # Not rendered into any manifest. All three are read out of the engine
    # command -- whichever flag that engine spells them in, see
    # `EngineSpec.flags_for_metric` -- because something outside the engine needs
    # a number the command already states: the pre-deploy capacity check and the
    # harness workload profile's context length for the first two, the router's
    # prefix-cache index for `blockSize`. A scenario states one only when the
    # command cannot (an entrypoint that carries the flag, or an engine with no
    # CLI flag for it at all). None means nobody said, and the dependent check is
    # skipped rather than run against a guess.
    maxModelLen: int | str | None = None
    blockSize: int | None = None
    gpuMemoryUtilization: float | None = Field(default=None, ge=0, le=1)

    # Computed at render time by RenderPlans._resolve_model_id_label and
    # injected for ${model.idLabel} template use -- not user-set.
    idLabel: str | None = None

    # Computed at render time by RenderPlans._resolve_model_hub_cache -- not
    # user-set. Under `uriProtocol: pvc+hf` the model volume holds a Hugging Face
    # hub cache, and this is the subdirectory of the volume it sits in. Relative
    # to the volume root on purpose: every container that touches the cache sees
    # that volume somewhere else -- the download Job at `downloadJob.mountPath`,
    # the serving pods where the modelservice chart mounts it, a standalone pod
    # at `standalone.modelMountPath`, the hostPath DaemonSet through the node
    # filesystem -- so each composes HF_HUB_CACHE from its own mount path and
    # this. None under every other protocol.
    hubCacheSubdir: str | None = None


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class HarnessResourcesConfig(BaseModel):
    """Harness pod resource configuration."""

    model_config = STRICT_CONFIG

    cpu: int | str
    memory: str
    memoryLimit: str | None = None
    """Memory limit; falls back to ``memory`` when unset."""


class InferencePerfConfig(BaseModel):
    """Inference-perf harness configuration."""

    model_config = STRICT_CONFIG

    rayonNumThreads: int


class HarnessConfig(BaseModel):
    """Benchmark harness configuration."""

    model_config = STRICT_CONFIG

    name: str
    profile: str | None = None
    experimentProfile: str | None = None
    executable: str
    # Optional pod-entrypoint override. step_07 reads harness.entrypoint
    # (default: the llm-d-benchmark.sh launcher); harnesses whose image has no
    # launcher in /usr/local/bin (e.g. eval-containers, which runs a standalone
    # eval image) point this at their script in the mounted scripts ConfigMap.
    entrypoint: str | None = None
    condaEnvName: str
    waitTimeout: int = Field(ge=0)
    loadParallelism: int = Field(ge=1)
    podLabel: str
    debug: bool
    resources: HarnessResourcesConfig
    nodeSelector: dict[str, str] = Field(default_factory=dict)
    tolerations: list[dict[str, Any]] = Field(default_factory=list)
    output: str
    inferencePerf: InferencePerfConfig
    namespace: str | None = None
    pvcSize: str | None = None
    # Cluster-specific overrides supplied via --cluster-config (deep-merged onto
    # the scenario). They render the harness pod's securityContext.runAsUser and
    # serviceAccountName (see 20_harness_pod.yaml.j2). Modeled here so a valid
    # cluster-config does not trip the extra="forbid" "Extra inputs" warning.
    # Type-only, no default: defaults.yaml remains the source of truth.
    runAsUser: int | None = None
    # Renders securityContext.privileged on the harness container. Independent
    # of runAsUser: privileged workloads usually also want runAsUser: 0, but
    # neither implies the other.
    privileged: bool | None = None
    serviceAccount: str | None = None


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class BenchmarkConfig(BaseModel):
    """Root validation model for the merged config dict.

    Uses ``extra="allow"`` at the root level so that sections not yet
    modeled are accepted without error.  This enables incremental adoption --
    only the explicitly modeled sections below are validated with
    ``extra="forbid"``.
    """

    model_config = ConfigDict(extra="allow")

    model: ModelConfig
    decode: DecodeConfig
    prefill: PrefillConfig
    engine: EngineCommonConfig
    harness: HarnessConfig
    parallelism: ParallelismConfig | None = None
    description: DescriptionConfig | None = None

    # Scenario-level workspace directory (equivalent to LLMDBENCH_CONTROL_WORK_DIR).
    # Used as workspace fallback when --workspace is not specified on the CLI.
    workDir: str | None = None


# ---------------------------------------------------------------------------
# Validation entry point
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def validate_config(
    merged_values: dict[str, Any],
    render_logger: Any | None = None,
) -> list[str]:
    """Validate a merged config dict against the benchmark schema.

    This function is intentionally **non-blocking**: it never raises
    exceptions.  All validation issues are collected and returned as a
    list of human-readable warning strings.

    Parameters
    ----------
    merged_values:
        The fully-merged config dict (defaults + scenario, after resolvers).
    render_logger:
        Optional logger instance (with ``log_warning`` method) from the
        rendering pipeline.  Falls back to the module-level stdlib logger.

    Returns
    -------
    list[str]
        Validation warning messages.  Empty list means the config is valid
        (within the scope of modeled sections).
    """
    warnings: list[str] = []

    try:
        BenchmarkConfig.model_validate(merged_values)
    except ValidationError as exc:
        for error in exc.errors():
            field_path = ".".join(str(loc) for loc in error["loc"])
            msg = f"Config validation: {field_path} -- {error['msg']}"
            warnings.append(msg)
            if render_logger and hasattr(render_logger, "log_warning"):
                render_logger.log_warning(msg)
            else:
                logger.warning(msg)
    except Exception as exc:
        msg = f"Config validation unexpected error: {exc}"
        warnings.append(msg)
        if render_logger and hasattr(render_logger, "log_warning"):
            render_logger.log_warning(msg)
        else:
            logger.warning(msg)

    return warnings
