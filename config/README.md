# Configuration

All declarative configuration for `llmdbenchmark` lives in this directory. The three subdirectories correspond to the inputs consumed by the plan phase rendering pipeline.

## Table of Contents

- [Directory Layout](#directory-layout)
- [How the Pieces Fit Together](#how-the-pieces-fit-together)
- [Config Override Chain](#config-override-chain)
  - [Method 1: Scenario File](#method-1-scenario-file-recommended-for-deployment-specific-config)
  - [Method 2: Environment Variables](#method-2-environment-variables-for-shellci-defaults)
  - [Method 3: CLI Arguments](#method-3-cli-arguments-highest-priority-runtime-overrides)
    - [Overriding arbitrary scenario keys (`--set`)](#overriding-arbitrary-scenario-keys---set)
  - [Method 4: Experiment Treatments](#method-4-experiment-treatments-for-parameter-sweeps)
- [Templates](#templates)
  - [Jinja2 Templates](#templatesjinja)
  - [defaults.yaml](#templatesvaluesdefaultsyaml)
- [KV Transfer and KV Events](#kv-transfer-and-kv-events)
- [Resource Configuration](#resource-configuration)
  - [Ephemeral Storage](#ephemeral-storage)
  - [Network Resources (RDMA/InfiniBand)](#network-resources-rdmainfiniband)
  - [Accelerator Resources](#accelerator-resources)
- [Affinity Configuration](#affinity-configuration)
- [Scenario Organization](#scenario-organization)
- [Engine Command](#engine-command)
- [Context-Length-Aware Routing](#context-length-aware-routing)
- [Init Containers](#init-containers)
- [Harness Entrypoint Configuration](#harness-entrypoint-configuration)
- [Flow Control Configuration](#flow-control-configuration)
- [Monitoring and Metrics](#monitoring-and-metrics)
- [KEDA Autoscaling](#keda-autoscaling)
  - [Generic KEDA ScaledObjects (`keda`)](#generic-keda-scaledobjects-keda)
- [Container Images](#container-images)
  - [Image Config Paths](#image-config-paths)
  - [Which Template Uses Which Image](#which-template-uses-which-image)
  - [Fallback Chains](#fallback-chains)
  - [Overriding Images](#overriding-images)
- [Scenarios](#scenarios)
  - [Guide Scenarios](#scenariosguides)
  - [Example Scenarios](#scenariosexamples)
  - [CI/CD Scenarios](#scenarioscicd)
  - [Creating a New Scenario](#creating-a-new-scenario)
- [Specifications](#specifications)
  - [Required Fields](#required-fields)
  - [Optional Fields](#optional-fields)
  - [Specification Auto-Discovery](#specification-auto-discovery)
  - [The base_dir Variable](#the-base_dir-variable)
  - [Creating a New Specification](#creating-a-new-specification)
  - [Naming and Collisions](#naming-and-collisions)
  - [Experiments](#experiments)
  - [Available Specifications](#available-specifications)
- [Usage](#usage)

---

## Directory Layout

```text
config/
    templates/
        jinja/                  Jinja2 templates that produce Kubernetes manifests
            _macros.j2          Shared macros (engine command, pod env, init containers)
            01_pvc_workload-pvc.yaml.j2    ... through 23_wva-namespace.yaml.j2
        values/
            defaults.yaml       Base configuration with all anchored defaults

    scenarios/                  Deployment overrides (merged on top of defaults)
        guides/                 Well-lit-path guide scenarios
        examples/               Minimal working examples (cpu, gpu, spyre)
        cicd/                   CI/CD pipeline environments

    specification/              Plan specifications (entry points for the CLI)
        guides/                 Well-lit-path guide specifications
        examples/               Minimal working specifications
        cicd/                   CI/CD pipeline specifications
```

## How the Pieces Fit Together

![Rendering Pipeline](../docs/images/rendering-pipeline.svg)

## Config Override Chain

Values are merged in a strict priority order during the plan phase. Later sources override earlier ones:

![Config Override Chain](../docs/images/config-override-chain.svg)

The merged result is written as `config.yaml` inside each rendered stack directory. This is the **single source of truth** for all step execution. Steps never define their own fallback defaults -- they read from `config.yaml` using `_require_config()` and raise a clear error if a required key is missing.

### How to Override Values

#### Method 1: Scenario File (recommended for deployment-specific config)

Create a scenario YAML under `config/scenarios/` that overrides only the values you need:

```yaml
scenario:
  - name: "my-deployment"
    common:
      model:
        name: Qwen/Qwen3-32B
      namespace:
        name: my-namespace
    modelservice:
      enabled: true
      decode:
        replicas: 4
    standalone:
      enabled: false
    fma:
      enabled: false
```

Only the keys you specify are overridden. Everything else comes from `defaults.yaml`.
The `common` section is inherited by every standup method. Method-specific
settings belong under `standalone`, `modelservice`, or `fma`.

`modelservice.common` is distinct from the stack-level `common` section: it
is passed through as the modelservice Helm chart's `common` values block.

**Multi-stack scenarios - the `shared:` block.** The scenario file also
accepts an optional top-level `shared:` key that's merged into every stack
before the per-stack overrides. Use it to lift scenario-wide config out of
duplicated per-stack blocks. Per-stack still wins, so any stack can override
a shared value:

```yaml
shared:
  modelservice:
    enabled: true
    gateway: { className: istio }
  httpRoute:
    mode: shared
    name: multi-model-route
    pathPrefix: /{stack.name}
    rewriteTo: /
  router:
    epp: { pluginsConfigFile: "optimized-baseline-plugins.yaml", ... }

scenario:
  - name: pool-a
    model: { name: Qwen/Qwen3-0.6B, ... }
    decode: { replicas: 1 }
  - name: pool-b
    model: { name: unsloth/Meta-Llama-3.1-8B, ... }
    decode: { replicas: 1 }
```

Render-time conveniences that activate only when `len(scenario) >= 2`:

- `downloadJob.name` and `router.monitoring.secretName` are auto-suffixed
  with each stack's `model_id_label` so parallel download Jobs and
  sibling router Helm releases don't collide. Explicit overrides
  (in `defaults.yaml`, `shared:`, or per-stack) are preserved.
- `storage.modelPvc.name` is **not** suffixed - every stack writes weights
  to a distinct `model.path` subdirectory on one shared PVC. This matches
  how NVMe / local-directory storage classes are typically deployed and
  lets cached weights be reused across runs without per-model duplication.
- When `httpRoute.mode: shared`, [08_httproute.yaml.j2](templates/jinja/08_httproute.yaml.j2)
  renders a single HTTPRoute in the first stack with one backendRef per
  sibling stack; other stacks render an empty file.
- Step 04 iterates `context.rendered_stacks` so every stack with
  a PVC-backed `modelservice.uriProtocol` (`pvc+hf`, `pvc`) or standalone runs a download Job
  against the shared PVC. Downloads run in parallel - total wall time
  ~ slowest model, not sum.

See [examples/multi-model-optimized-baseline.yaml](scenarios/examples/multi-model-optimized-baseline.yaml)
for a complete example and the developer guide's [Multi-Stack Scenarios](../docs/developer-guide.md#multi-stack-scenarios-and-the-shared-block)
section for the merge semantics.

**Shared role config - the `roleDefaults:` block.** `shared:` lifts config out
of duplicated *stacks*; `roleDefaults:` does the same for duplicated *roles*
inside one stack. In most deployments `decode` and `prefill` are the same pod --
same container, same probes, same volumes, same environment -- and only the
launch command really differs. Write the shared shape once:

```yaml
scenario:
  - name: pd
    modelservice:
      enabled: true

      roleDefaults:
        resources:
          limits: { memory: 128Gi, cpu: "32" }
          requests: { memory: 128Gi, cpu: "32" }
        extraEnvVars:
          - name: NCCL_DEBUG
            value: "WARN"

      prefill:
        enabled: true
        engine:
          command: |
            vllm serve Qwen/Qwen3-32B --port 8000 --tensor-parallel-size 1

      decode:
        replicas: 2
        parallelism: { tensor: 2 }        # decode is wider
        resources:                        # ...and needs less memory per replica
          limits: { memory: 64Gi, cpu: "16" }
          requests: { memory: 64Gi, cpu: "16" }
        engine:
          command: |
            vllm serve Qwen/Qwen3-32B --port 8200 --tensor-parallel-size 2
```

The rules:

- Precedence is `defaults.yaml` < `common` < `roleDefaults` < the role's own
  block. Naming a key under `decode:` always wins, per leaf -- `decode` above
  replaces both memory values and keeps the shared `extraEnvVars`.
- It seeds `decode`, `prefill` and `standalone`, and only where that role
  actually models the key: `shm` reaches `decode` alone, so putting it in
  `roleDefaults` does not invent an `shm` on `prefill`. `nok8s` is not a role
  in this sense and is never seeded -- it describes an ssh connection, not a pod.
- A key no role models at all is a typo and fails the render, naming what could
  have been written instead.
- Values replace rather than combine, lists included. A role that names
  `extraEnvVars` replaces the shared list instead of appending to it, so
  plumbing every role must keep belongs in `roleDefaults` only.
- It may be written at stack level or under `modelservice:` -- wherever the role
  blocks themselves are. A stack with no `roleDefaults` is unaffected.

Only roles a scenario actually mentions are seeded: `roleDefaults` never brings
a role into existence, so it cannot turn a disabled `prefill` on.

**Example: GPU scenario with a custom vLLM image**

A role pins its own server image under `<role>.engine.image` -- the same key on
every role (`decode`, `prefill`, `standalone`, `nok8s`). Whatever a role leaves
out is filled in from `images.<engine>`, picked by the engine the role's command
launches. See [Container Images](#container-images) for the full image config
reference.

```yaml
scenario:
  - name: "gpu-custom-vllm"
    standalone:
      enabled: true
      engine:
        image:
          repository: docker.io/vllm/vllm-openai
          tag: v0.8.5
      replicas: 1
      parallelism:
        tensor: 1
    namespace:
      name: my-gpu-ns
```

```bash
llmdbenchmark --spec gpu standup -c config/scenarios/my-gpu-custom.yaml
```

For modelservice deployments (e.g. `guides/optimized-baseline`), override `images.vllm` instead:

```yaml
scenario:
  - name: "ms-custom-vllm"
    images:
      vllm:
        repository: ghcr.io/llm-d/llm-d-cuda
        tag: v0.5.0
```

The deployed image is recorded in the `llm-d-benchmark-standup-parameters` ConfigMap for audit.

#### Method 2: Environment Variables (for shell/CI defaults)

Export `LLMDBENCH_*` environment variables to set defaults without passing CLI flags every time. Env vars override scenario/defaults values but are themselves overridden by explicit CLI flags.

```bash
# Set common defaults in .bashrc or CI pipeline
export LLMDBENCH_SPEC=guides/optimized-baseline
export LLMDBENCH_NAMESPACE=my-team-ns
export LLMDBENCH_KUBECONFIG=~/.kube/my-cluster
export LLMDBENCH_DRY_RUN=true

# Now run without repeating flags
llmdbenchmark standup
llmdbenchmark standup -p override-ns   # CLI -p wins over LLMDBENCH_NAMESPACE
```

Boolean env vars accept `1`, `true`, or `yes` (case-insensitive). See the [CLI Reference](../README.md#cli-reference) for the full mapping of flags to env var names. Active overrides are logged at startup.

#### Method 3: CLI Arguments (highest priority, runtime overrides)

CLI arguments override both defaults, scenario values, and environment variables:

```bash
# Override namespace
llmdbenchmark --spec my-spec.yaml.j2 standup -p my-namespace

# Override deployment method (standalone, modelservice, fma, kustomize, nok8s)
llmdbenchmark --spec my-spec.yaml.j2 standup -t standalone

# Override model
llmdbenchmark --spec my-spec.yaml.j2 standup -m "meta-llama/Llama-3.1-8B"

# Override Helm release name
llmdbenchmark --spec my-spec.yaml.j2 standup -r my-release

# Combine multiple overrides
llmdbenchmark --spec my-spec.yaml.j2 standup -p my-ns -t modelservice -r my-release
```

##### Overriding arbitrary scenario keys (`--set`)

The flags above cover the values that have a dedicated flag. **Any** key in
the merged config can be overridden with `--set`, using the same dotted
paths a scenario file uses. This is what makes a
near-duplicate scenario file unnecessary:

```bash
# One key -- the SGLang flavour of a guide, no second scenario file
llmdbenchmark --spec guides/optimized-baseline standup \
  -t kustomize -o kustomize.acceleratorBackend=gpu/sglang

# Several keys: comma-separated, or repeat the flag
llmdbenchmark --spec guides/pd-disaggregation standup \
  --set 'decode.replicas=2,prefill.replicas=4' \
  --set 'storage.modelPvc.size=2Ti'
```

Values are parsed as YAML, so `4`, `true`, `[a, b]` and `{x: 1}` mean what
they would in the scenario file. Commas inside `[]`, `{}` or quotes belong to
the value, not the separator. Because it is real YAML, `012` is octal 10 and
`1:30` is 90 -- quote the value (`"foo='012'"`) to keep it a string; a
warning is emitted whenever a value is read as something other than it looks.

Multi-line values are folded onto one line: real newlines in a plain scalar
collapse into spaces, which silently changes the meaning of a shell command.
Wrap the value in double quotes so `\n` is an escape --
`--set 'decode.engine.command="export FOO=1\nvllm serve /model-cache/x"'`
-- or, for a full multi-line engine command, set it in the scenario file
instead; `--set` is best suited to single-line values.

Lists are assigned whole, never indexed: `engine.volumeMounts.0.name=x`
is rejected (it would silently replace the entire list) -- pass the full
list instead, `'engine.volumeMounts=[{name: x, mountPath: /x}]'`.

In a multi-stack scenario, prefix a key with a stack name (or an fnmatch
glob) to scope it; unprefixed applies to every stack:

```bash
# Different value per pool, both still deployed
llmdbenchmark --spec examples/multi-model-optimized-baseline standup \
  --set 'qwen3-06b:decode.replicas=4,llama-31-8b:decode.replicas=1'

# A common floor with one exception (exact name beats the global)
llmdbenchmark --spec examples/multi-model-optimized-baseline standup \
  --set 'decode.resources.limits.memory=64Gi' \
  --set 'llama-31-8b:decode.resources.limits.memory=32Gi'
```

A selector matching no stack is a hard error, not a silent no-op. Every
applied override is logged with its previous value
(`[llama-31-8b] Scenario override: decode.replicas: 1 -> 4`).

`--set` is available on every subcommand that renders templates
(`plan`, `standup`, `smoketest`, `run`, `teardown`, `experiment`) -- pass it
to each phase of a lifecycle, since they all re-render. Full reference:
[docs/standup.md](../docs/standup.md#overriding-scenario-values-from-the-cli---set).

> [!IMPORTANT]
> `--set` always means the **scenario**. On `run` and `experiment`,
> `-o/--overrides` is a **different** flag that overrides the workload
> profile; the two can be combined. `standup` has no workload profile, so
> it accepts `--set` only.

There is also `--cluster-config FILE`, which takes the same overrides as a
YAML mapping for values that are constant per cluster (storage class,
service account). `--set` wins over that file on a contested key. See
[docs/openshift-setup.md](../docs/openshift-setup.md).

#### Method 4: Experiment Treatments (for parameter sweeps)

Specification files can define experiments with setup and run treatments that generate multiple stacks with different parameter values:

```yaml
experiments:
  - name: "replica-sweep"
    attributes:
      - name: "setup"
        factors:
          - name: decode.replicas
            levels: [1, 2, 4]
        treatments:
          - decode.replicas: 1
          - decode.replicas: 2
          - decode.replicas: 4
```

Each treatment produces a separate rendered stack, enabling parallel deployment and comparison.

---

## Templates

### `templates/jinja/`

Jinja2 templates that produce Kubernetes resource definitions. Each template corresponds to a specific infrastructure component:

| Template | Output |
|----------|--------|
| `01_pvc_workload-pvc.yaml.j2` | Workload PVC for harness data |
| `02_pvc_model-pvc.yaml.j2` | Model storage PVC |
| `03_cluster-monitoring-config.yaml.j2` | OpenShift workload monitoring config |
| `04_download_job.yaml.j2` | Model download Job |
| `05_namespace_sa_rbac_secret.yaml.j2` | Namespace, ServiceAccount, RBAC, secrets |
| `06_pod_access_to_harness_data.yaml.j2` | Harness data access pod |
| `07_service_access_to_harness_data.yaml.j2` | Harness data access service |
| `08_httproute.yaml.j2` | HTTPRoute for inference gateway |
| `09_helmfile-gateway-provider.yaml.j2` | Helmfile for gateway provider (Istio/agentgateway) |
| `10_helmfile-main.yaml.j2` | Main helmfile (llm-d-infra, modelservice) |
| `11_infra.yaml.j2` | Infrastructure chart values |
| `12_router-values.yaml.j2` | llm-d router (EPP + InferencePool) Helm values |
| `13_ms-values.yaml.j2` | Modelservice Helm values |
| `14_standalone-deployment_yaml.j2` | Standalone vLLM Deployment |
| `15_standalone-service_yaml.j2` | Standalone vLLM Service |
| `16_pvc_extra-pvc.yaml.j2` | Extra PVCs (e.g., scratch space) |
| `17_standalone-podmonitor.yaml.j2` | Standalone PodMonitor for metrics |
| `18_podmonitor.yaml.j2` | Modelservice PodMonitor for metrics |
| `19_wva-kustomize.yaml.j2` | Workload Variant Autoscaler kustomize wrapper |
| `20_harness_pod.yaml.j2` | Benchmark harness pod |
| `21_prometheus-adapter-values.yaml.j2` | Prometheus adapter values |
| `22_prometheus-rbac.yaml.j2` | Prometheus RBAC resources |
| `23_wva-namespace.yaml.j2` | WVA namespace resources |
| `31_nok8s-epp-config.yaml.j2` | No-Kubernetes EPP (file-discovery) config |
| `32_nok8s-epp-endpoints.yaml.j2` | No-Kubernetes EPP endpoints (worker list) |
| `33_nok8s-envoy.yaml.j2` | No-Kubernetes Envoy bootstrap |
| `34_nok8s-containers.yaml.j2` | No-Kubernetes container launch spec (see [nok8s](../docs/nok8s.md)) |
| `_macros.j2` | Shared Jinja2 macros (engine command, pod env, init containers) |

Templates use Jinja2 conditionals to skip rendering when their feature is disabled. For example, standalone templates only render when `standalone.enabled` is `true` (and the `nok8s` templates only when `nok8s.enabled` is `true`). Steps check for empty rendered files via `_has_yaml_content()` and skip applying them.

### `templates/values/defaults.yaml`

The base configuration file containing every configurable parameter with sensible defaults. Uses YAML anchors extensively for DRY references across sections.

**Key sections:**

| Section | Purpose |
|---------|---------|
| `_anchors` | Reusable YAML anchors for ports, resources, probes, parallelism |
| `model` | Model identifiers, paths, cache settings |
| `namespace` | Deploy and harness namespace names |
| `release` | Helm release name prefix |
| `gateway` | Gateway class and provider configuration |
| `serviceAccount` | Service account name and configuration |
| `huggingface` | HuggingFace token, secret name, and enabled flag |
| `storage` | PVC sizes, storage class, download settings |
| `decode` | Decode pod configuration (replicas, resources, `engine.command`) |
| `prefill` | Prefill pod configuration (disabled by default) |
| `standalone` | Standalone deployment settings (disabled by default) |
| `modelservice` | Modelservice deployment settings (enabled by default) |
| `nok8s` | No-Kubernetes deployment settings (disabled by default) -- see [nok8s](../docs/nok8s.md) |
| `images` | Container image repositories, tags, and pull policies |
| `engine` | Engine-neutral pod shape shared by every role (shell, service port, preprocess script, volumes, pull secret) -- no engine parameters, see [Engine Command](#engine-command) |
| `harness` | Benchmark harness configuration |
| `wva` | Workload Variant Autoscaler settings |
| `keda` | Generic KEDA ScaledObjects with configurable Prometheus auth (any cluster) |
| `control` | Context secret name |
| `lws` | LeaderWorkerSet configuration |
| `agentgateway` | agentgateway provider configuration |
| `router` | llm-d-router chart values (EPP, tokenizer, inferencePool, monitoring) |

**YAML anchors:** The file uses anchors (`&name`) and aliases (`*name`) to ensure consistency. For example, `&service_port` is defined once as `8000` and referenced by `engine.servicePort` and `routing.servicePort`. Note that the port an *engine* binds is not an anchor -- it comes from the `--port` in that role's `engine.command`.

## HuggingFace Configuration

The `huggingface` section controls authentication for downloading models from HuggingFace Hub.

| Field | Type | Default | Description |
|---|---|---|---|
| `huggingface.enabled` | `bool` | `true` | Enable HuggingFace authentication. Auto-set to `false` at render time when no token is found |
| `huggingface.token` | `str` | `""` | HuggingFace API token (typically set via `HF_TOKEN` or `LLMDBENCH_HF_TOKEN` env var) |
| `huggingface.secretName` | `str` | `hf-token` | Name of the Kubernetes secret storing the token |
| `huggingface.secretKey` | `str` | `HF_TOKEN` | Key within the secret |

When `huggingface.enabled` is `false`, the following are skipped:
- HuggingFace token secret creation in the model namespace
- `secretKeyRef` mount for `HF_TOKEN` on the download job/DaemonSet (`hf download` reads `HF_TOKEN` from the environment directly, no `hf auth login` step is run)
- `secretKeyRef` mounts for `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` on vLLM and harness pods
- `authSecretName` on the ModelService CR

This allows public models (e.g. `facebook/opt-125m`) to be deployed without a token. Gated models (e.g. `meta-llama/Llama-3.1-8B`) require a valid token and will fail at the model access check if one is not provided.

The `enabled` flag is auto-computed during plan rendering by `_resolve_hf_token()` in `render_plans.py`. It checks `HF_TOKEN`, `LLMDBENCH_HF_TOKEN`, and the scenario YAML in that order.

## Config Variable Substitution

Almost nothing in a scenario needs this. An engine command states its flags and
their values literally -- that is the whole point of it being verbatim -- so the
numbers and the model id are written out, and llm-d-benchmark reads back off the
command the few it needs. `${dotted.path}` exists for the cases that shape
cannot cover:

- **One value, two places, and one of them is not a command.** An EPP plugin
  config or an `InferenceObjective` needs a number the command already states.
  Written twice they can silently disagree.
- **One command shared by several stacks that serve different models.** Each
  stack states its own `model.name`; the shared command names the stack.
- **A value llm-d-benchmark derives, which a scenario therefore cannot write.**
  `model.idLabel` is a `{first8}-{sha8}-{last8}` hash of the model id.
- **A stack with no engine command at all.** Fast model actuation runs the
  launcher process as the engine, so there is no `vllm serve` line for the
  capacity check and the workload profile to read their numbers off. Those
  stacks state `model.maxModelLen` / `blockSize` / `gpuMemoryUtilization` in
  the `model:` block and `fma.launcher.options` references them, which is the
  first case with the direction reversed: the non-command field is the source.

If a reference does not fall into one of those, write the value out instead.

### Syntax

`${section.key}` resolves against the merged config (defaults + scenario) at
render time. The path must contain at least one dot -- that is what separates a
config reference from a container or shell variable, which is never touched.

Any scalar in the merged config can be referenced, but the list of ones worth
referencing is short: `${model.idLabel}` and `${namespace.name}` (derived),
`${model.name}` (multi-stack only), and a capacity number the command already
states and a non-command field needs to match.

### Where to use

- `extraEnvVars` values
- `pluginsCustomConfig` -- inline EPP plugin configuration
- `extraObjects` -- inline Kubernetes manifests
- `<role>.engine.command` / `engine.args` -- multi-stack scenarios only

### Examples

All three are in the tree.

An EPP plugin needs the engine's page size, and it is not a command
([precise-prefix-cache-routing.yaml](scenarios/guides/precise-prefix-cache-routing.yaml)):

```yaml
    decode:
      engine:
        command: |
          vllm serve meta-llama/Llama-3.1-8B-Instruct \
          --port 8200 \
          --block-size 64            # <- the one statement of the page size

    modelservice:
      router:
        epp:
          pluginsCustomConfig:
            plugins.yaml: |
              - type: precise-prefix-cache-producer
                parameters:
                  tokenProcessorConfig:
                    # Must equal the engine's page size: this reconstructs block
                    # hashes on the engine's boundaries. Read off `--block-size`
                    # above, so there is one number and it is the engine's own.
                    blockSizeTokens: ${model.blockSize}
```

An `InferenceObjective` has to name the router pool, whose name is derived
([turn-priority-fairness.yaml](scenarios/guides/turn-priority-fairness.yaml)):

```yaml
      extraObjects:
        - apiVersion: llm-d.ai/v1alpha2
          kind: InferenceObjective
          spec:
            priority: -1
            poolRef:
              # `idLabel` is a hash of the model id that llm-d-benchmark
              # computes, so this is the only way to write it.
              name: ${model.idLabel}-router
```

One command, several models
([multi-model-optimized-baseline.yaml](scenarios/examples/multi-model-optimized-baseline.yaml)):

```yaml
modelservice:
  decode:
    engine:
      # Every flag is written out. Only the serve target varies, because the
      # stacks below serve different models and share this one command.
      command: |
        vllm serve ${model.name} \
        --port 8200 \
        --block-size 64 \
        --max-model-len 8192

scenario:
  - name: "qwen3-06b"
    model:
      name: Qwen/Qwen3-0.6B
  - name: "llama-31-8b"
    model:
      name: unsloth/Meta-Llama-3.1-8B
```

### Container and shell variables

There is nothing to configure. A command is passed through as written, so a `$`
is just a character in it: `$(POD_IP)` is expanded by Kubernetes, `$MAX_NUM_SEQS`
and `${LWS_WORKER_INDEX:-0}` by the shell that runs the command. The dot rule
above is what keeps them out of scope.

A scenario reaches for one when the value is not knowable until the pod is
running -- the pod's own IP in a KV-events topic
([precise-prefix-cache-routing.yaml](scenarios/guides/precise-prefix-cache-routing.yaml)),
or a number that has to differ between replicas of one Deployment (see
[Context-Length-Aware Routing](#context-length-aware-routing)).

### Behavior

- Substitution runs after all resolvers (model, namespace, version, etc.) so all values are available.
- If a reference cannot be resolved, it is left as-is and a warning is logged.
- Non-string values (integers, booleans) are converted to strings when embedded.
- The original config dict is not mutated - a deep copy is used.

## Model Artifact Protocol (`modelservice.uriProtocol`)

Controls how the modelservice Helm chart locates and loads model weights. Set via `modelservice.uriProtocol` in your scenario or defaults.

| Protocol | `modelArtifacts.uri` generated | PVC | Download Job | What the engine command names |
|---|---|---|---|---|
| `pvc+hf` (default) | `pvc+hf://<modelPvc.name>/<model.path>` | Yes | Yes (stages an HF hub cache) | the plain model id -- `vllm serve Qwen/Qwen3-32B` |
| `pvc` | `pvc://<modelPvc.name>/<model.path>` | Yes | Yes (flat weights directory) | the mounted path, plus `--served-model-name <id>` |
| `hf` | `hf://<model.huggingfaceId>` | No | No | the plain model id |

### How it works

**`pvc+hf://` (default).** The PVC holds a Hugging Face *hub cache*
(`models--<org>--<model>/snapshots/<sha>/...`), not a flat copy of one model's
files:

1. Step 04 creates the PVC (`storage.modelPvc`) and launches a download Job
   ([04_download_job.yaml.j2](templates/jinja/04_download_job.yaml.j2)) that runs
   `hf download` with `HF_HUB_CACHE` pointed into the PVC, so the cache layout is
   what lands there.
2. Template 13 generates `modelArtifacts.uri: pvc+hf://<pvc-name>/<model.path>`.
   `model.path` must end in the two segments of the model id
   (`models/Qwen/Qwen3-32B`): the chart reads the model argument off those last
   two segments, and everything between the claim name and them is the cache
   directory it points `HF_HUB_CACHE` at.
3. The chart mounts the PVC and sets `HF_HUB_CACHE` in the serving container, so
   the plain model id in the launch command resolves against the staged bytes
   instead of being re-pulled from the Hub.
4. The model mount is writable under this protocol (`modelArtifacts.readOnly:
   false`) because resolving an id through `huggingface_hub` takes a lock and
   refreshes `refs/<revision>` inside the cache.

This is the protocol to use by default: the weights are pre-staged, startup is
fast, and the launch command is the same line you would run on a node.

**`pvc://`.** The PVC holds a plain directory of one model's files. The download
Job stages it with `--local-dir`, the chart mounts it read-only, and the engine
is given the *path* -- so this is the one protocol where the command also needs
`--served-model-name <id>` to advertise something clients can ask for. Use it for
weights that never came from the Hub in the first place (a converted or
locally-built checkpoint); see
[experimental/kimi-k3-h100.yaml](scenarios/experimental/kimi-k3-h100.yaml).

**`hf://`.** No PVC and no download Job: the chart sets `HF_HOME` on the mount
path and the engine pulls from the Hub at pod start. Same plain model id in the
command as `pvc+hf`. Useful for CI/CD, quick testing, or a cluster where storage
cannot be provisioned (`--no-pvc` forces this protocol). For gated models
`huggingface.secretName` is passed as `authSecretName` so the chart can
authenticate -- which it does for every protocol, not just this one.

### Scenario example

```yaml
scenario:
  - name: "my-hf-deploy"
    decode:
      engine:
        command: |
          vllm serve facebook/opt-125m --port 8200
    modelservice:
      enabled: true
      uriProtocol: hf     # No PVC, no download job - fetch at runtime
```

### Code path

1. `llmdbenchmark/standup/steps/step_04_model_namespace.py` - `_requires_pvc_download()` returns `True` for any `pvc`-prefixed protocol
2. `llmdbenchmark/parser/render_plans.py` - `_resolve_model_hub_cache()` derives `model.hubCacheSubdir` (the cache's directory *relative to the model volume*, so each container composes `HF_HUB_CACHE` from its own mount path) under `pvc+hf`; `_validate_model_uri()` checks `model.path` against the protocol that has to interpret it
3. `config/templates/jinja/13_ms-values.yaml.j2` - generates the `hf://`, `pvc://` or `pvc+hf://` URI
4. `config/templates/jinja/04_download_job.yaml.j2` (and `03_download_daemonset.yaml.j2` for hostPath) - stages either layout; only rendered when the protocol is PVC-backed

## Chart Versions

All Helm chart and component versions are centralized in the `chartVersions` section of `defaults.yaml`. This is the single place to bump versions when upgrading components.

| Field | Default | Description |
|---|---|---|
| `chartVersions.istioBase` | `1.29.1` | Istio base chart version |
| `chartVersions.istiod` | `1.29.1` | Istiod chart version (also used as gateway version) |
| `chartVersions.llmDInfra` | `auto` | llm-d-infra Helm chart (auto-resolved via helm) |
| `chartVersions.llmDModelservice` | `v0.4.16` | llm-d-modelservice Helm chart version |
| `chartVersions.inferencePool` | `v1.3.0` | Inference pool chart version |
| `chartVersions.wva` | `auto` | Workload Variant Autoscaler chart (auto-resolved) |
| `chartVersions.agentgateway` | `v2.2.3` | agentgateway chart version |
| `chartVersions.lws` | `v0.11.0` | LeaderWorkerSet chart version |

Versions set to `auto` are resolved at plan time by `VersionResolver` using `helm search repo` or OCI registry queries. Fixed versions are used as-is.

### Overriding versions in a scenario

Add a `chartVersions` section to your scenario YAML. Only include the versions you want to change - the rest inherit from defaults:

```yaml
scenario:
  - name: "my-upgrade-test"
    chartVersions:
      llmDModelservice: "0.5.0"    # pin to specific version
      agentgateway: "v2.3.0"        # upgrade agentgateway
```

### Pinning all versions for reproducibility

To ensure a benchmark run is fully reproducible, pin every `auto` version to a specific value. Run `plan` first to see what `auto` resolves to, then copy those values into your scenario:

```yaml
scenario:
  - name: "reproducible-bench"
    chartVersions:
      llmDInfra: "v1.4.0"          # was auto
      llmDModelservice: "v0.4.9"   # was auto
      wva: "0.5.1"                 # was auto
```

### Upgrading Istio

Istio uses two charts (`istio-base` and `istiod`) that must be the same version. Override both:

```yaml
chartVersions:
  istioBase: "1.30.0"
  istiod: "1.30.0"
```

### How `auto` resolution works

1. For charts with a `helmRepositories` entry: queries the repo via `helm search repo` or OCI registry
2. Falls back to the registry's tag list API for OCI registries
3. Selects the latest semver-compatible tag
4. Resolved versions are logged during `plan`: `📦 Resolved chart llmDInfra to v1.4.0 (via repo URL)`

> **Note:** `auto` versions may change between runs as upstream charts release new versions. Pin versions in your scenario for consistent results across runs.

## KV Transfer and KV Events

KV cache transfer is an engine parameter, so it belongs in the role's
`engine.command` and nowhere else. vLLM spells it `--kv-transfer-config` with a
JSON payload; other engines spell it differently, and the schema moves between
releases. Write the line exactly as the llm-d guides write it and it reaches the
container unchanged:

```yaml
decode:
  engine:
    command: |
      vllm serve Qwen/Qwen3-32B \
      --port 8200 \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

Anything the payload needs that is only known at render time or at pod start is
reachable without a configuration key of its own -- a name the plan assigns
(the router Service, the namespace) or a fact the node supplies (the pod's own
IP). Engine *behaviour* is not one of these: a flag whose value differs by
hardware is written out, literally, in a scenario for that hardware
([examples/intel-xpu.yaml](scenarios/examples/intel-xpu.yaml),
[examples/spyre.yaml](scenarios/examples/spyre.yaml),
[examples/cpu.yaml](scenarios/examples/cpu.yaml)). A substitution point per
difference is how a command stops saying what the engine will do.

| Reference | Resolved by | Works without a shell? | Example |
|---|---|---|---|
| `${dotted.path}` | llm-d-benchmark, at render time | yes -- it is gone before the pod exists | `${model.idLabel}`, `${namespace.name}` |
| `$(VAR)` | the kubelet, expanding `args` from the container's own env | yes | `$(POD_IP)`, `$(ENGINE_PORT)`, `$(MODEL_NAME)`, and anything in `extraEnvVars` |
| `$VAR` | the shell the command runs under | **no** | same variables, but only where `modelCommand` gives the container a shell |

Only the first row is llm-d-benchmark's own: the substitution regex requires at
least one dot precisely so the other two pass through untouched.

Prefer `$(VAR)` to `$VAR`. Under `modelCommand: imageDefault` -- the distroless
path, no shell in the image -- `$(VAR)` still expands and `$VAR` reaches the
engine as those literal characters. And on the shell path the failure is worse
than silent: a `$(VAR)` the kubelet cannot resolve is left as-is, and the shell
then reads it as command substitution.

So a NIXL connector, and a ZMQ event publisher whose topic has to identify the
publishing pod, both come out of one command line:

```yaml
decode:
  extraEnvVars:
    - name: KV_EVENTS_ENDPOINT
      value: "tcp://*:5556"
  engine:
    command: |
      vllm serve Qwen/Qwen3-32B \
      --port 8200 \
      --prefix-caching-hash-algo sha256_cbor \
      --kv-transfer-config '{"kv_connector":"NixlConnector", "kv_role":"kv_both"}' \
      --kv-events-config '{"enable_kv_cache_events":true, "publisher":"zmq", "endpoint":"$(KV_EVENTS_ENDPOINT)", "topic":"kv@$(POD_IP):$(ENGINE_PORT)@$(MODEL_NAME)"}'
```

The events endpoint is the address the engine *binds* and publishes on -- the
router subscribes to each pod it discovers -- so it is a wildcard bind, not a
Service name.

Working examples, each copied from the corresponding llm-d guide:

| Scenario | Connector |
|---|---|
| [guides/pd-disaggregation.yaml](scenarios/guides/pd-disaggregation.yaml) | `NixlConnector`, on both the decode and prefill commands |
| [guides/tiered-prefix-cache.yaml](scenarios/guides/tiered-prefix-cache.yaml) | `OffloadingConnector` with `kv_connector_extra_config` |
| [guides/wide-ep.yaml](scenarios/guides/wide-ep.yaml) | `NixlConnector` with `kv_load_failure_policy` |
| [guides/precise-prefix-cache-routing.yaml](scenarios/guides/precise-prefix-cache-routing.yaml) | `--kv-events-config` (no transfer connector) |

The one KV-related value llm-d-benchmark still needs for itself is the EPP-side
page size, because the router's token processor hashes prefixes on the same
block boundaries the engine writes and a value that disagrees scores silently
against the wrong blocks. It is not configured: the command already carries the
page size as an ordinary engine flag (vLLM `--block-size`, SGLang `--page-size`),
and that flag is read back onto `model.blockSize` for the router to use -- see
[What is read back out of the command](#what-is-read-back-out-of-the-command).
State `model.blockSize` yourself only when there is no flag to read: TRT-LLM has
no CLI spelling for it at all (it lives in the `--extra_llm_api_options` YAML),
and an image whose entrypoint carries the flag has no command here to read.

---

## Resource Configuration

Pod resource limits and requests are configured per role (decode/prefill) in the scenario YAML. The template reads `memory` and `cpu` from `resources.limits` and `resources.requests`, but some resource types use dedicated fields.

### Ephemeral Storage

Ephemeral storage is configured via a dedicated field, **not** inside the `resources` block:

```yaml
engine:
  ephemeralStorage: 1Ti    # applied to every role
```

Or per role:

```yaml
decode:
  ephemeralStorage: 1Ti
prefill:
  ephemeralStorage: 500Gi
```

The resource name used in the rendered output is controlled by `engine.ephemeralStorageResource` (default: `ephemeral-storage`). Values placed in `resources.limits.ephemeral-storage` are **not** read by the template and will be silently ignored.

### Network Resources (RDMA/InfiniBand)

Network resources for high-performance interconnects are configured via:

```yaml
engine:
  networkResource: "auto"   # auto-detect from cluster nodes
  networkNr: "1"            # number of network devices
```

When `networkResource` is set to `"auto"`, the `ClusterResourceResolver` queries the cluster nodes at render time and discovers available RDMA/IB resources (e.g., `rdma/roce_gdr`, `rdma/ib`). The resolved resource name and count are then rendered into the pod resource limits and requests.

**Requirements:** Auto-detection requires cluster connectivity. Running `plan` without a cluster when `networkResource: "auto"` is set will fail with a clear error. Scenarios that don't need RDMA/IB should not set this field (the default is empty).

### Accelerator Resources

Two separate settings control GPU/accelerator behavior:

- **`accelerator.count`** -- how many accelerator devices (e.g., GPUs) the pod requests in its Kubernetes resource limits
- **`parallelism.tensor`** -- how many parallel workers for tensor operations, passed as `--tensor-parallel-size` to vLLM

These are independent values that are often the same but can differ:

| Scenario | accelerator.count | parallelism.tensor | Why they differ |
|----------|------------------|--------------------|-----------------|
| Standard GPU | 2 (or unset) | 2 | Same -- 2 GPUs, 2 tensor parallel workers |
| CPU-only | 0 | 2 | No GPUs, but TP=2 uses CPU threads |
| Spyre | 1 | 4 | 1 device supports 4 tensor parallel ranks |
| Expert parallel | unset | 1 | GPUs come from DP_local, not TP |

#### How accelerator count is resolved

When `accelerator.count` is **not set**, it defaults to `parallelism.tensor`. This matches the most common case where each tensor parallel rank needs its own GPU.

When `accelerator.count` is **explicitly set to 0**, no accelerator resources are added to the pod limits. This is required for CPU-only scenarios that use tensor parallelism for CPU threads.

The resolution order for standalone deployments is:
1. `standalone.accelerator.count` (if set)
2. Top-level `accelerator.count` (if set)
3. `standalone.parallelism.tensor` (fallback)

For modelservice deployments:
1. `decode.accelerator.count` (if set)
2. `decode.parallelism.tensor` (fallback)

#### Accelerator resource name

The Kubernetes resource name (e.g., `nvidia.com/gpu`, `ibm.com/spyre_vf`) is configured separately:

```yaml
accelerator:
  type: nvidia               # accelerator family
  resource: "nvidia.com/gpu"  # Kubernetes resource name (set to "auto" for cluster detection)
```

Per-role overrides are supported:

```yaml
decode:
  accelerator:
    resource: ibm.com/spyre_vf  # override for non-NVIDIA accelerators
    count: 1                     # explicit device count
```

When `resource` is set to `"auto"`, the cluster resource resolver detects the accelerator resource name from cluster nodes at plan time (requires cluster connectivity).

---

## Model ID Label

Kubernetes resource names are derived from the model ID using a hashed `model_id_label` format:

```
{first8}-{sha256_8}-{last8}
```

For example, `meta-llama/Llama-3.1-8B` produces `meta-lla-a1b2c3d4-a-3-1-8b`. This format keeps resource names within the 63-character DNS label limit while remaining identifiable. The label is computed automatically by `_resolve_model_id_label` during plan rendering -- scenarios do not need to set it manually. The `model_id_label` field replaces the former `model.shortName` in all templates and Python code.

The hashing matches the bash `model_attribute()` function so that CLI tools and rendered manifests produce consistent names.

---

## Affinity Configuration

Node affinity controls which cluster nodes pods are scheduled on. Configured under the top-level `affinity` section in `defaults.yaml` or per scenario.

#### Modes

| Mode | Config | Behavior |
|------|--------|----------|
| **Disabled** (default) | `affinity.enabled: false` or omit entirely | Pods schedule on any available node |
| **Explicit** | `affinity.enabled: true` with `nodeSelector` labels | Pods only schedule on nodes matching the specified labels |
| **Auto** | Pass `--affinity auto` via CLI or set `LLMDBENCH_AFFINITY=auto` | Auto-detects GPU/accelerator labels from cluster nodes at render time |

#### Explicit example

```yaml
affinity:
  enabled: true
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
```

#### Optional pod affinity/anti-affinity

The `affinity` section also supports `podAffinity` and `podAntiAffinity` rules for co-locating or spreading pods across nodes:

```yaml
affinity:
  enabled: true
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          topologyKey: kubernetes.io/hostname
```

---

## Scenario Organization

Each stack is organized into four YAML sections:

- **`common`** -- settings inherited by every deployment method (model,
  namespace, storage, engine, harness, images, and workDir)
- **`standalone`** -- settings used when `standalone.enabled: true`
- **`modelservice`** -- settings used when `modelservice.enabled: true`,
  including decode, prefill, gateway, router, routing, httpRoute,
  inferenceExtension, and multinode
- **`fma`** -- settings used when `fma.enabled: true`

The renderer expands `common` and method-specific sub-sections to the flat
effective `config.yaml` consumed internally. A key written flat at the
scenario root also resolves; if both spellings are present, the sectioned one
wins.
Treatment and CLI overrides are applied afterward and therefore take highest
precedence.

Each scenario must set exactly one of `standalone.enabled: true` or `modelservice.enabled: true`. Only one deployment method can be active at a time. The CLI `-t` flag overrides the scenario value (e.g., `-t standalone` forces standalone even if the scenario says modelservice). If both are passed via CLI, a warning is logged and modelservice is used. Templates use Jinja2 conditionals to skip rendering when the corresponding flag is `false`.

Workload settings such as `workDir` and `harness` belong under `common` because
they apply regardless of the selected standup method.

---

## Engine Command

llm-d-benchmark is engine-agnostic. It does not model an engine's command line,
does not generate one, and has no configuration key for any engine parameter.
A role states the launch command **verbatim** -- what you would type on a node,
copied in unchanged:

```yaml
decode:
  engine:
    command: |
      vllm serve meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 8200 \
      --tensor-parallel-size 4 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.95
```

That is the whole model configuration too. The id in the command is where the
model is named, and `model.name`, `model.huggingfaceId`, the PVC path and the
pod labels are read off it -- see
[Referring to the model](#referring-to-the-model-and-the-cluster-from-inside-the-command)
for the one thing a command cannot say and the two cases that want a variable
instead of the literal id.

Switching engines is a different command and nothing else -- the same
scenario shape, no new keys:

```yaml
decode:
  engine:
    command: |
      python3 -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 8200 \
      --tp-size 4 \
      --context-length 32768 \
      --mem-fraction-static 0.9
```

[examples/engines.yaml](scenarios/examples/engines.yaml) (smoke-sized, one
0.6B decode pod) and
[guides/optimized-baseline.yaml](scenarios/guides/optimized-baseline.yaml) (guide
scale) each carry all three commands, the vLLM one active and the SGLang and
TensorRT-LLM ones commented out directly beneath it: strip the `# ` prefix from
one block, comment out the other, and the rendered plan changes in two places --
the launch line and the image the launcher selects. They are also where the
exception is written down, because TensorRT-LLM is the one engine that needs
three keys beyond the command (`model.blockSize`, `monitoring.metricsPath` and
`LD_LIBRARY_PATH`, each commented in place next to the key it belongs to).

Each commented group is tagged `# @engine <name>`, which is what keeps it from
rotting. Nothing parses a comment, so an alternative would otherwise be the one
block in the file that no test ever reads. The tag also means you do not have to
make the edit by hand -- `--engine <name>` makes it for you:

```bash
llmdbenchmark standup --spec examples/engines --engine sglang -p "$NS"
llmdbenchmark standup --spec examples/engines --engine trtllm -p "$NS"
util/scenario-inventory.py --alternative trtllm    # which scenarios offer it
util/test-scenarios.sh --plan --engine trtllm      # render every one of them switched in
```

`--engine <name>` uncomments the tagged groups into a copy under the workspace --
the repo is not touched -- drops the definitions they replace, and renders that
copy. Asking for the engine the scenario already launches is a logged no-op;
asking for one it neither launches nor offers is an error, because only the file
knows which companion keys move with the command.
`tests/test_engine_alternatives.py` asserts the same switch in process, which is
what enforces "all four blocks or none" for TensorRT-LLM: drop one and a capacity
number arrives empty, which the test fails on.

This is why there is no `sglang.yaml` and no `trtllm.yaml`. A per-engine file is
the same stack as its vLLM sibling with one string changed, which the tagged
group already expresses -- and the two files then drift apart with nothing
comparing them. Adding a fourth engine here costs one commented block, not a
fourth file.

### What is read back out of the command

The command text is authoritative and is never rewritten. It is *read* once,
for the handful of facts Kubernetes must know before the process starts --
because they decide the shape of objects created around the engine, not the
engine's own behaviour:

| Read | Why it cannot wait for the engine | Landed on |
|---|---|---|
| The launcher (`vllm serve`, `python3 -m sglang.launch_server`, `trtllm-serve`, `llm-d-inference-sim`) | Picks the server image, health path and metrics path | `<role>.engine.name`, `<role>.engine.image` |
| `--port` | Sizes the container port, the probes and the routing sidecar's upstream | `<role>.engine.port` |
| The model reference (`--model`, `--model-path`, the positional, or `--served-model-name`) | Names what the PVC stages, what the pod labels and the HTTPRoute match, and what the harness sends requests to | `model.name`, `model.huggingfaceId`, `model.path`, `model.idLabel` |
| The capacity pair: context length and memory fraction | Feed the pre-deploy capacity check, which sizes KV cache before anything is created | `model.maxModelLen`, `model.gpuMemoryUtilization` |
| The KV page size (vLLM `--block-size`, SGLang `--page-size`) | The router's prefix-cache index must hash on the same block boundaries the engine writes | `model.blockSize` |

That is the whole list -- five facts, and `--port` is spelled identically by
every supported engine, so it costs nothing per engine. Every other flag is
opaque and reaches the container untouched: batch widths, expert parallelism, KV
connector configuration, parallelism widths. There is no list of engine
parameters to keep up to date, and no parameter to re-learn before writing a
scenario.

**A device count is not read out of the command.** The kubelet grants
accelerators before the engine process exists, so the count is stated in
Kubernetes' vocabulary -- `<role>.resources.limits.<accelerator resource>`, or
the `accelerator.count` / `<role>.parallelism` shorthands -- and has to agree
with the width the command gives the engine. A role running `--tensor-parallel-size 2`
writes the width twice on purpose: once for the engine, once for the chart.
Inferring it from a product of flag widths would mean tracking every engine's
spelling of every width, and would disagree with the pod spec the moment one of
them changed.

**Where a read and a stated value disagree, the command wins** -- and the
override is reported as a warning. The command is the text handed to the engine,
so it is the only one of the two that is certainly true; a consumer told the
other number would be sizing KV cache the engine never allocates, or hashing
pages it never writes. A `model.*` value stated in the scenario is therefore a
*fallback* for a command that says nothing, not an override of one that does --
which is what lets a scenario state its context length once, in the flag the
engine actually reads, instead of twice. A value that had to be overruled is a
scenario asking for something it will not get, so it is surfaced rather than
silently dropped.

When several roles run, the value is taken from the first of `decode`,
`standalone`, `nok8s`, `prefill` that supplies it, skipping any role that is
disabled or scaled to zero.

Each engine's flag spellings live in one place,
[`llmdbenchmark/engine/spec.py`](../llmdbenchmark/engine/spec.py). Adding an
engine means adding one `EngineSpec` there -- not a template branch, not a
configuration section.

### Referring to the model and the cluster from inside the command

**Write the model id.** A command that names its model literally is read as the
statement of which model the plan serves: `model.name`, `model.huggingfaceId`,
`model.path` and `model.idLabel` are all derived from it, so a scenario needs no
`model:` block to serve a model, and there is no second place that can disagree
with the engine. A scenario that *does* state `model.name` keeps it, and a
command naming a different model is reported rather than patched.

One model key does not come from the command. `model.shortName` prefixes this
stack's Deployments, Services and PVCs; a model id is a fact the command states,
but a short name is a choice, and its derived form is a
`{first8}-{sha8}-{last8}` hash. Scenarios write down a readable one
(`shortName: qwen-qwen3-32b`) and it is left alone.

**No `--served-model-name`.** The id in the command is also the id clients ask
for: every engine here advertises whatever it was told to serve when nothing
says otherwise (vLLM's `get_served_model_name`, SGLang's `served_model_name`
defaulting to `model_path`). So the flag is redundant under the default
`pvc+hf` and under `hf`, where the serve target is already the id. It earns its
place only under `uriProtocol: pvc`, where the serve target is a mounted
directory and something has to name the API -- that is the one protocol where the
id is read off `--served-model-name` instead.

One case wants a variable rather than the literal id: **one command shared by
several stacks that serve different models.** `${model.name}` resolves against
each stack's own merged config, which is what
[examples/multi-model-optimized-baseline.yaml](scenarios/examples/multi-model-optimized-baseline.yaml)
uses -- one `shared:` command serving stacks that differ only in which model they
load. The model facts run the other way there: the `model:` blocks are
authoritative and the command follows them.

Anything else a command might want to reference is covered by
[Config Variable Substitution](#config-variable-substitution) -- which is mostly
a list of reasons not to.

### Adding a few words to a shared command

One stack, one command, written out in full -- that is the normal case, and a
per-role difference means moving the `engine:` block down into the role. The
exception is a scenario deploying several stacks off one `shared:` command where
a stack or two differ in a flag. Writing that out means repeating sixteen
identical lines to change one of them, and the copies drift. `engine.extraArgs`
is a list of words appended to the end of the command, unexamined:

```yaml
shared:
  modelservice:
    decode:
      engine:
        command: |
          vllm serve ${model.name} \
          --port 8200 \
          --max-model-len 8192 \
          --gpu-memory-utilization 0.95 \
          --block-size 64

scenario:
  - model:
      name: Qwen/Qwen3-0.6B                 # runs the shared line as written
  - model:
      name: meta-llama/Llama-3.1-8B-Instruct
    modelservice:
      decode:
        engine:
          extraArgs: ["--max-model-len", "4096"]
```

The llama stack runs the shared line with `--max-model-len 4096` on the end.
**Repeating a flag the command already carries overrides it**, because every
engine's argument parser keeps the last occurrence -- and so does the reader
described above, so `model.maxModelLen` for that stack reads back as 4096, the
number the engine will actually use. Nothing here knows what `--max-model-len`
means; two strings are joined with a space.

Three things to know about it:

* **It appends, it never replaces.** The overridden flag is still visible in the
  rendered Deployment, stated twice. That is deliberate: removing the earlier
  occurrence would mean knowing that `--max-model-len` takes a value while
  `--enable-prefix-caching` does not -- per flag, per engine, which is the
  parameter modelling this design exists to avoid.
* **It cannot change the serve target.** The words land at the end, so the
  positional model argument is out of reach; a shared command still writes
  `${model.name}`.
* **A role's own list replaces a plan-wide one** rather than adding to it, the
  same way `command` does -- so one place states the words for a role and there is
  no question of what order two lists concatenate in.

A width is the usual reason to reach for this, and a width is also the case that
is not finished by the flag alone: `["--tensor-parallel-size", "2"]` tells the
engine, and `parallelism.tensor` plus the accelerator count tell the chart. See
[What is read back out of the command](#what-is-read-back-out-of-the-command).

### Declaring the engine

Normally nothing declares the engine: the launcher in the command identifies
it. `engine.name` (plan-wide) or `<role>.engine.name` exists for the two cases
where the command cannot say:

- **A role with no command**, because the image's own entrypoint starts the
  server (`command: ""`, see below). There is no launcher to read, so the
  declaration is what picks the image and the health path.
- **A wrapper script**, where the engine is launched by something like
  `/opt/app-root/spyre_entrypoint.sh`. The wrapper matches no launcher
  signature, so declaring the engine is what makes the flags after it readable
  in that engine's spelling -- see
  [examples/spyre.yaml](scenarios/examples/spyre.yaml).

A declaration is checked against what the command actually launches, and a
mismatch is reported. Known engines: `vllm`, `sglang`, `trtllm`, `sim`.

### Running the image's own entrypoint

`command: ""` means "this role has no launch line of its own":

```yaml
decode:
  engine:
    command: ""
    args: ["--model", "/model-cache/${model.path}", "--port", "8000"]
```

The rendered container then carries `args:` with **no** `command:`, so a
distroless image with no shell still runs, and the port falls back to the
engine's default. Note that `command: null` does **not** do this -- the merge
skips null, so the inherited default command survives. See
[cicd/kind.yaml](scenarios/cicd/kind.yaml), which runs `llm-d-inference-sim`
this way.

### Which port to bind

Two ports are involved, and only one of them is an engine parameter:

- **`engine.servicePort`** (8000) is infrastructure -- the port the Service and
  the gateway expose. It is not passed to any engine.
- **The port in the command's `--port`** is what the engine binds inside the
  container. It is read from the command; `<role>.engine.port` overrides it for
  a role whose port cannot be read (a fixed entrypoint), and the two
  disagreeing is a warning.

Which value to write depends on who else is on the pod, and getting it wrong is
a stack that comes up green and answers nothing, so it is an **error**, not a
warning:

| Role | Bind |
|---|---|
| Decode, routing sidecar enabled (the default) | **8200** -- the sidecar owns 8000 and forwards upstream |
| Decode, `routing.proxy.enabled: false` (including `gateway.className: none`) | **8000** -- nothing bridges, so the engine must answer on the Service port |
| Prefill | **8000** -- prefill pods never get a sidecar, and the decode sidecar reaches them there for P/D |
| Standalone | **8000** |

Probe ports follow the engine's bind port automatically, and can be overridden
with `<role>.probes.startup.port`, `<role>.probes.liveness.port` and
`<role>.probes.readiness.port`.

### Preprocess script

One step runs in the same container before the engine, chained onto the front
of the command: `<role>.engine.preprocessCommand`, else `engine.preprocessScript`,
else `/bin/true`. It is chained with `;` so the engine starts regardless --
except under `contextLengthRanges`, where the preprocess also labels the pod and
an unlabeled pod is invisible to the context-length-aware scorer; there it is
chained with `&&` so a failure actually stops the container.

Nothing else is prepended or appended. The engine's command line is
byte-for-byte what you wrote.

---

## Context-Length-Aware Routing

This section explains how to configure per-pod context-length-aware routing. This feature allows different decode (or prefill) pods to handle different context length ranges, enabling the [llm-d inference scheduler](https://github.com/llm-d/llm-d-inference-scheduler) to route requests to the most appropriate pod based on token count.

### How It Works

The setup has three layers:

1. **Scenario YAML** -- you define `contextLengthRanges` per role (decode/prefill), and, where the replicas must differ in more than a label, `extraEnvVars` whose values are `,,`-delimited (one entry per pod) and which the role's `engine.command` references
2. **Template rendering** -- the benchmark tool converts these lists into environment variables injected into pods:
   - `LLMDBENCH_POD_LABELS` for pod self-labeling (format: `label_eq_value,label_eq_value`)
   - each `,,`-delimited `extraEnvVars` value, verbatim, for the script to split
3. **Pod startup** -- the preprocess script (`set_llmdbench_environment.py`) runs inside each pod, extracts the pod's index from its hostname (via LeaderWorkerSet), selects the correct variant values, re-exports the env vars, and self-labels the pod using pykube

The inference scheduler's `context-length-aware` plugin then reads the `llm-d.ai/context-length-range` label from each pod and routes requests accordingly.

### Prerequisites

- **Multinode (LeaderWorkerSet) must be enabled.** The per-pod index is derived from sequential pod names assigned by LWS (e.g., `decode-0`, `decode-1`). Without LWS, pods get random Deployment hash suffixes and the index cannot be determined.
- **Kubeconfig secret must be mounted.** Pods need K8s API access to self-label. This is handled by mounting the `llmdbench-context` secret (created automatically by step 04).
- **Preprocess script must be configured.** The `engine.preprocessScript` must run `set_llmdbench_environment.py` and source the generated env file.

### Scenario Configuration

Add the following to your scenario YAML under the `decode` (or `prefill`) section:

```yaml
scenario:
  - name: "my-context-aware-deployment"

    # Enable multinode -- REQUIRED for per-pod variants
    multinode:
      enabled: true

    modelservice:
      enabled: true

    decode:
      replicas: 2

      # Per-pod context-length-range labels.
      # List length must match replicas.
      # Each pod gets the label at its index (pod-0 gets first, pod-1 gets second).
      contextLengthRanges:
        - "0-8000"
        - "8000-32768"

      # Per-replica values (optional). A `,,`-delimited value means "one
      # entry per pod": the preprocess script picks the entry at this pod's
      # index and re-exports the variable holding just that value. The number
      # of entries must match replicas.
      #
      # These variable names are the scenario's own -- nothing in
      # llm-d-benchmark knows them. They match only because the command below
      # references them, which is what makes this work for any engine and any
      # parameter, not a fixed list of supported keys.
      extraEnvVars:
        - name: MAX_MODEL_LEN
          value: "8000,,32768"
        - name: MAX_NUM_SEQS
          value: "64,,16"

      # This is the one place a command does NOT write its numbers out, and the
      # reason is Kubernetes, not configuration: these two replicas are one
      # Deployment, so they share one pod spec and one command string, and the
      # only thing that tells pod-0 from pod-1 at runtime is its ordinal. A
      # value that must differ per pod therefore cannot be a literal. Write out
      # every flag that does NOT vary -- only the ones that do become variables.
      #
      # `$MAX_MODEL_LEN`, not `$(MAX_MODEL_LEN)`: the `$(...)` form is expanded
      # by Kubernetes when the pod is created, which would substitute the whole
      # unsplit "8000,,32768". Only the shell, reading what the preprocess
      # script wrote, sees this pod's entry.
      engine:
        command: |
          vllm serve meta-llama/Llama-3.1-8B-Instruct \
          --port 8200 \
          --tensor-parallel-size 4 \
          --max-model-len $MAX_MODEL_LEN \
          --max-num-seqs $MAX_NUM_SEQS

      parallelism:
        tensor: 4
        data: 1
        dataLocal: 1
        workers: 1

    # Configure the router EPP with the context-length-aware plugin.
    # `apiVersion: llm-d.ai/v1alpha1` is the canonical API group on the
    # llm-d-router charts; the legacy
    # `inference.networking.x-k8s.io/v1alpha1` is still accepted but
    # deprecated.
    #
    # `router.tokenizer.enabled: true` adds the chart's `vllm-render` sidecar
    # to the EPP pod, serving vLLM's /render endpoints over loopback HTTP on
    # `router.tokenizer.port` (default 8000); `token-producer` points at it.
    # `router.tokenizer.modelName` is filled from `model.name` automatically.
    router:
      tokenizer:
        enabled: true
      epp:
        pluginsConfigFile: "context-length-aware-config.yaml"
        pluginsCustomConfig:
          context-length-aware-config.yaml: |
            apiVersion: llm-d.ai/v1alpha1
            kind: EndpointPickerConfig
            plugins:
              - type: token-producer
                parameters:
                  modelName: "${model.name}"
                  vllm:
                    url: http://localhost:8000
              - type: context-length-aware
                parameters:
                  label: llm-d.ai/context-length-range
                  enableFiltering: true
            schedulingProfiles:
              - name: default
                plugins:
                  - pluginRef: token-producer
                  - pluginRef: context-length-aware

    # Preprocess script and kubeconfig secret volume are required
    engine:
      preprocessScript: "python3 /setup/preprocess/set_llmdbench_environment.py && source $HOME/llmdbench_env.sh"
      volumes:
        - name: k8s-llmdbench-context
          type: secret
          secret:
            secretName: llmdbench-context
      volumeMounts:
        - name: k8s-llmdbench-context
          mountPath: /etc/kubeconfig
          readOnly: true
```

### What Gets Generated

Given the configuration above, the rendered pod template will contain:

```yaml
env:
  - name: MAX_MODEL_LEN
    value: "8000,,32768"
  - name: MAX_NUM_SEQS
    value: "64,,16"
  - name: LLMDBENCH_POD_LABELS
    value: "llm-d.ai/context-length-range_eq_0-8000,llm-d.ai/context-length-range_eq_8000-32768"
  - name: LLMDBENCH_POD_LABELS_REQUIRED
    value: "true"
```

At pod startup, the preprocess script:
- **Pod decode-0**: re-exports `MAX_MODEL_LEN=8000`, `MAX_NUM_SEQS=64`, labels itself with `llm-d.ai/context-length-range=0-8000`
- **Pod decode-1**: re-exports `MAX_MODEL_LEN=32768`, `MAX_NUM_SEQS=16`, labels itself with `llm-d.ai/context-length-range=8000-32768`

The engine then reads those variables from its own command line, so the two
pods run the same command text with different sizes. Because the labels are
load-bearing -- the scorer selects pods by them -- `LLMDBENCH_POD_LABELS_REQUIRED`
is set and the preprocess step is chained with `&&`: a pod that cannot label
itself does not come up serving traffic it would answer outside its range.

### Standalone Deployments

Context-length-aware routing is **not applicable** to standalone deployments. Standalone mode has no router EPP or routing layer, so there is nothing to route requests based on context length. The `contextLengthRanges` and `router.epp` fields only apply to the modelservice deployment path.

### Verifying the Setup

After standup, verify the labels are applied:

```bash
kubectl get pods -l llm-d.ai/role=decode -n <namespace> --show-labels
```

You should see each pod with a distinct `llm-d.ai/context-length-range` label.

Check the EPP logs for context-length-aware plugin activation:

```bash
kubectl logs <epp-pod> -c epp -n <namespace> | grep -i "context-length"
```

### Preprocess Script

The preprocess script (`set_llmdbench_environment.py`) is required when using `contextLengthRanges` or any per-replica env var. It runs inside each pod at startup and performs two tasks:

1. **Splits `,,`-delimited env vars** -- a value like `MAX_MODEL_LEN="8000,,32768"` is split by the pod's LWS index and re-exported, so each pod gets its own value. Any variable name works; the command is what gives it meaning.
2. **Self-labels pods** -- applies the `llm-d.ai/context-length-range` label to each pod via the K8s API, so the inference scheduler can route requests based on context length.

The script requires:
- `engine.preprocessScript` set to run the script and source the env file
- The `preprocesses` ConfigMap volume mounted at `/setup/preprocess`
- The `llmdbench-context` secret volume mounted at `/etc/kubeconfig` (for K8s API access)

See the commented-out sections in the example scenarios for the exact configuration.

### Reference

- [llm-d inference scheduler architecture: context-length-aware](https://github.com/llm-d/llm-d-inference-scheduler/blob/main/docs/architecture.md#contextlengthaware)
- [GPU example scenario](scenarios/examples/gpu.yaml) -- contains a worked `contextLengthRanges` + per-replica `extraEnvVars` pair and commented-out `router.epp` configuration
- [Spyre example scenario](scenarios/examples/spyre.yaml) -- the same, with Spyre-specific volumes

---

## Init Containers

Init containers run before the model-server container to perform environment setup tasks such as network configuration (RDMA/InfiniBand route tables), hardware detection, and environment variable preparation.

#### How it works

1. The init container runs the benchmark image with `set_llmdbench_environment.py -i` (init container mode)
2. It writes environment configuration to `/shared-config/llmdbench_env.sh` on a shared emptyDir volume
3. The model-server container sources this file via `preprocessScript: "source /shared-config/llmdbench_env.sh"`

The `shared-config` emptyDir volume and volumeMount are configured per scenario under `engine.volumes` and `engine.volumeMounts` (`defaults.yaml` defines no volumes of its own).

#### Image resolution

Init container images can be specified three ways, in order of preference:

1. **`imageKey: <entry>`** -- references an entry under `images.*` in `defaults.yaml` (e.g. `imageKey: benchmark`, `imageKey: udsTokenizer`). The resolver expands it to `<repo>:<tag>` and inherits `imagePullPolicy` from the same entry. This is the recommended form -- single source of truth, automatically tracks version bumps in `defaults.yaml`.

2. **`image: <full-string>`** -- any image string. Tags ending in `:auto` are resolved against the registry's tag list at render time (falling back to `podman`). Use for one-off images that don't have an `images.*` entry.

3. **Neither set** -- the template falls back to `images.benchmark`.

Setting both `image` and `imageKey` on the same entry is a config error and aborts plan generation. An unknown `imageKey` (no matching `images.*` entry) does too -- the error message lists the available keys.

The resolved image is visible in the rendered `ms-values.yaml` in the workspace directory.

#### Environment variable propagation

Init containers receive the same environment variables as the model-server container. This is handled by the `build_ms_env_vars()` macro in `_macros.j2`, which is called for both the model-server container and each init container that does not already define its own `env:` section.

The propagated env vars include all core vLLM configuration (ports, model parameters, parallelism), NCCL/UCX transport settings, pod metadata (POD_IP, namespace), and any scenario-specific `extraEnvVars` (NCCL_EXCLUDE_IB_HCA, FLEX_*, etc.). This ensures preprocess scripts have full access to the deployment configuration for RDMA/HCA detection, NCCL tuning, and other runtime setup.

If an init container defines its own `env:` section in the scenario YAML, the automatic injection is skipped -- the scenario's explicit env vars take precedence.

#### Scenario configuration

Init containers are configured per scenario (the default in `defaults.yaml` is `initContainers: []`). Each guide scenario explicitly defines the preprocess init container:

```yaml
decode:
  initContainers:
    - name: preprocess
      imageKey: benchmark  # -> images.benchmark.{repository,tag} from defaults.yaml
      imagePullPolicy: Always
      command: ["set_llmdbench_environment.py", "-e", "/shared-config/llmdbench_env.sh", "-i"]
      securityContext:
        capabilities:
          add:
            - IPC_LOCK
            - SYS_RAWIO
      volumeMounts:
        - name: shared-config
          mountPath: /shared-config
```

The `securityContext` capabilities vary by scenario:
- `IPC_LOCK` and `SYS_RAWIO` are the base capabilities needed for most deployments
- `NET_ADMIN` and `NET_RAW` are additionally required for scenarios that need network configuration (route tables, InfiniBand detection) --e.g., `precise-prefix-cache-routing` and `tiered-prefix-cache`
- Scenarios like `optimized-baseline` and `pd-disaggregation` use only the base capabilities

For scenarios with prefill pods (e.g., `pd-disaggregation`, `wide-ep`), add the same block under the `prefill` section as well.

#### Custom preprocessing

To use a different preprocessing script, change the `command` and/or `image`:

```yaml
decode:
  initContainers:
    - name: preprocess
      image: my-registry/my-init:v1.0  # custom image --used as-is, no auto-resolution
      command: ["my-setup-script.sh", "-o", "/shared-config/llmdbench_env.sh"]
      volumeMounts:
        - name: shared-config
          mountPath: /shared-config
```

The script must write a sourceable shell file to `/shared-config/llmdbench_env.sh` --the main container's `preprocessScript` sources it on startup.

#### Adding additional init containers

```yaml
decode:
  initContainers:
    - name: preprocess
      imageKey: benchmark
      command: ["set_llmdbench_environment.py", "-e", "/shared-config/llmdbench_env.sh", "-i"]
      volumeMounts:
        - name: shared-config
          mountPath: /shared-config
    - name: my-custom-init
      image: my-registry/my-init:latest  # one-off image; no `images.*` entry needed
      command: ["my-setup-script.sh"]
```

#### Disabling init containers

Omit the `initContainers` field or leave it as the default (`[]`).

---

## Harness Entrypoint Configuration

The harness entrypoint is the shell script executed inside the harness pod to orchestrate benchmark execution, metrics collection, and in-container analysis. By default, the entrypoint is `llm-d-benchmark.sh`, but it can be overridden per scenario.

#### Configuration

The entrypoint is read from `harness.entrypoint` in the plan config (which comes from `defaults.yaml` or a scenario override). If not set, it defaults to `llm-d-benchmark.sh`.

```yaml
harness:
  entrypoint: llm-d-benchmark.sh  # default
```

#### Custom entrypoints

To use a different entrypoint script (e.g., for a custom harness or specialized workflow):

```yaml
harness:
  entrypoint: my-custom-entrypoint.sh
```

The custom script must be present in the harness container image. It receives the same environment variables as the default entrypoint (`LLMDBENCH_*` variables, kubeconfig, namespace, results directory, etc.).

#### What the default entrypoint does

The `llm-d-benchmark.sh` entrypoint handles:

1. **Kubeconfig setup** -- Configures cluster access inside the pod using the base64-encoded context from `LLMDBENCH_BASE64_CONTEXT_CONTENTS`
2. **Pre-benchmark metrics scrape** -- Collects baseline vLLM metrics before the benchmark starts
3. **Harness execution** -- Runs the selected harness script (e.g., `inference-perf-llm-d-benchmark.sh`) with retry logic
4. **Post-benchmark metrics scrape** -- Collects final vLLM metrics after the benchmark completes
5. **In-container analysis** -- Runs analyzer scripts to produce benchmark reports

---

## Flow Control Configuration

Flow control is an EPP (inference scheduler) feature that manages request queuing and load distribution. When enabled, the EPP buffers requests based on pool capacity rather than sending them immediately to pods.

#### Enabling flow control

Flow control is configured through the EPP plugin configuration. The specific plugin config file is set in the scenario YAML:

```yaml
router:
  epp:
    pluginsConfigFile: flow-control-config.yaml  # name of the plugin config
```

#### Monitoring flow control

When flow control is active, additional Prometheus metrics are emitted by the EPP pod (see [Monitoring and Metrics](#monitoring-and-metrics) below for the full list). To scrape these metrics, enable EPP monitoring:

```yaml
router:
  monitoring:
    prometheus:
      enabled: true
    interval: "10s"
```

#### Using flow control in experiments

To compare performance with and without flow control, define setup treatments that vary the plugin configuration:

```yaml
setup:
  factors:
    - LLMDBENCH_VLLM_MODELSERVICE_GAIE_PLUGINS_CONFIGFILE
  levels:
    LLMDBENCH_VLLM_MODELSERVICE_GAIE_PLUGINS_CONFIGFILE: "default,flow-control-config"
  treatments:
    - "default"
    - "flow-control-config"
```

---

## Monitoring and Metrics

The benchmark supports Prometheus-based monitoring at three levels: global monitoring configuration, per-deployment PodMonitors, and EPP (inference scheduler) metrics.

#### Global monitoring settings

Configured under the top-level `monitoring` section in `defaults.yaml`:

| Field | Default | Description |
|---|---|---|
| `monitoring.enabled` | `true` | Enable monitoring infrastructure |
| `monitoring.enableUserWorkload` | `true` | Enable OpenShift user workload monitoring |
| `monitoring.podmonitor.enabled` | `true` | Create PodMonitor resources for Prometheus scraping |
| `monitoring.metricsPath` | `/metrics` | Prometheus scrape path |
| `monitoring.scrapeInterval` | `"30s"` | Prometheus scrape interval |
| `monitoring.timeSeriesMetrics` | See `defaults.yaml` | Metrics retained for processing, time-series graphs, and benchmark-report observability. Custom Prometheus metrics can be added without code changes. |
| `monitoring.installPrometheusCrds` | `false` | Install Prometheus CRDs (PodMonitor, ServiceMonitor) during standup. Required for clusters without Prometheus Operator (e.g. Kind). |

When `monitoring.enabled` is `true` and running on OpenShift, the `03_cluster-monitoring-config.yaml.j2` template renders a ConfigMap to enable user workload monitoring.

#### Per-deployment PodMonitors

Decode and prefill sections have their own `monitoring.podmonitor` config that controls PodMonitor creation:

```yaml
decode:
  monitoring:
    podmonitor:
      enabled: true
      portName: "metrics"
      path: "/metrics"
      interval: "30s"
      labels: {}
      annotations: {}
      relabelings: []
      metricRelabelings: []
```

When `podmonitor.enabled: true`, the templates `17_standalone-podmonitor.yaml.j2` (standalone) or `18_podmonitor.yaml.j2` (modelservice) render PodMonitor CRDs that tell Prometheus to scrape vLLM pods.

**Key metrics exposed by vLLM pods** (scraped via PodMonitor):
- `vllm:kv_cache_usage_perc` -- KV cache utilization (%)
- `vllm:gpu_cache_usage_perc` / `vllm:cpu_cache_usage_perc` -- GPU/CPU cache utilization (%)
- `vllm:gpu_memory_usage_bytes` / `vllm:cpu_memory_usage_bytes` -- memory usage
- `vllm:num_requests_running` -- active requests in batch
- `vllm:num_requests_waiting` -- queued requests
- `vllm:num_requests_swapped` -- requests swapped to CPU
- `vllm:num_preemptions_total` -- cumulative preemptions
- `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` -- prefix cache hit rate
- `vllm:external_prefix_cache_hits_total` / `vllm:external_prefix_cache_queries_total` -- cross-instance cache
- `vllm:nixl_xfer_time_seconds` / `vllm:nixl_bytes_transferred` -- NIXL KV transfer metrics

See [metrics_collection.md](../docs/metrics_collection.md) for the full list of collected metrics.

#### EPP (Inference Scheduler) monitoring

The router EPP has its own monitoring config under `router.monitoring`:

```yaml
router:
  monitoring:
    secretName: inference-gateway-sa-metrics-reader-secret
    interval: "10s"
    prometheus:
      enabled: true
      auth:
        enabled: true
```

This creates a ServiceMonitor for the EPP pod, enabling Prometheus to scrape inference scheduler metrics:

**Pool-level gauges:**
- `inference_pool_average_kv_cache_utilization` -- pool-wide KV cache utilization (%)
- `inference_pool_average_queue_size` -- average request queue depth
- `inference_pool_average_running_requests` -- average running requests
- `inference_pool_ready_pods` -- ready pod count

**Scheduler and request histograms:**
- `inference_extension_scheduler_e2e_duration_seconds` -- end-to-end scheduling latency
- `inference_extension_plugin_duration_seconds` -- per-plugin processing time
- `inference_extension_request_duration_seconds` -- total request duration
- `inference_extension_request_ttft_duration_seconds` -- time to first token

**Token and routing metrics:**
- `inference_extension_input_tokens` / `inference_extension_output_tokens` -- token distributions
- `inference_extension_normalized_time_per_output_token` -- NTPOT distribution
- `inference_extension_prefix_indexer_hit_ratio` / `inference_extension_prefix_indexer_size` -- prefix indexer

**P/D decision metrics:**
- `llm_d_inference_scheduler_pd_decision_total` -- P/D routing decisions
- `llm_d_inference_scheduler_disagg_decision_total` -- disaggregation decisions

When flow control is enabled (see [KV Transfer and KV Events](#kv-transfer-and-kv-events) for EPP config), additional metrics are emitted:
- `inference_extension_flow_control_queue_size` -- flow control queue depth
- `inference_extension_flow_control_pool_saturation` -- pool saturation level
- `llm_d_epp_flow_control_pool_saturation` -- pool saturation (0.0–1.0); KEDA
  EPP-saturation scale trigger (`saturationThreshold`)
- `llm_d_epp_request_running` -- running requests; KEDA EPP-saturation scale
  trigger (`runningRequestsThreshold`)

#### CLI monitoring flags (`--monitoring` / `--no-monitoring`)

The `--monitoring` and `--no-monitoring` flags control monitoring across both standup and run phases. These are tri-state: when neither is passed, the scenario defaults apply unchanged.

**`--monitoring` (standup):**
- Ensures PodMonitor resources are created for Prometheus scraping of vLLM pods
- Sets EPP verbosity to 4 (richer logs for post-run analysis)

**`--monitoring` (run):**
- Sets `metricsScrapeEnabled: true` — harness pods run `collect_metrics.sh` to scrape `/metrics` from all vLLM pods during the benchmark
- After each treatment, captures model-serving, EPP, and IGW pod logs
- Runs `process_epp_logs.py` on captured EPP logs to extract scheduling metrics

**`--no-monitoring` (standup):**
- Disables PodMonitor creation (`monitoring.podmonitor.enabled: false`)
- Disables router ServiceMonitor creation (`router.monitoring.prometheus.enabled: false`)
- Use this when the cluster lacks Prometheus CRDs (PodMonitor, ServiceMonitor) and you want to avoid CRD-not-found errors during Helm install

**No flag passed:**
- Scenario defaults from `defaults.yaml` apply as-is
- By default, `monitoring.podmonitor.enabled: true` (PodMonitors are created at standup)
- By default, `monitoring.metricsScrapeEnabled: false` (harness does not scrape metrics during run)
- To enable metrics scraping during run, pass `--monitoring` explicitly

**Environment variable:** `LLMDBENCH_MONITORING=true` or `LLMDBENCH_MONITORING=false` can be used as an alternative to the CLI flags. The CLI flag takes precedence when both are set.

#### Enabling monitoring in a scenario

To enable PodMonitor-based metrics collection for a deployment:

```yaml
scenario:
  - name: "my-monitored-deployment"
    monitoring:
      podmonitor:
        enabled: true
    decode:
      monitoring:
        podmonitor:
          enabled: true
```

#### Benchmark report integration

The analysis pipeline converts collected results into v0.2 benchmark reports (`benchmark-report/llmd_benchmark_report/`). Reports include:
- **Performance metrics**: TTFT, TPOT, ITL, request latency, throughput
- **Resource metrics**: KV cache usage, GPU/CPU memory, GPU utilization
- **Time series data**: Per-interval metric snapshots

Reports are generated in both YAML and JSON formats. See `benchmark-report/README.md` for the full schema reference.

#### Prometheus adapter (for autoscaling)

The `21_prometheus-adapter-values.yaml.j2` template configures a Prometheus adapter that bridges WVA (Workload Variant Autoscaler) metrics to the Kubernetes external metrics API. This is only needed when using WVA-based autoscaling.

---

## KEDA Autoscaling

### Generic KEDA ScaledObjects (`keda`)

Renders one or more `ScaledObject` resources from a user-defined list. Works on any Kubernetes cluster. KEDA must already be installed in the cluster.

**Templates rendered:** `27_keda-scaledobjects.yaml.j2`, `27a_keda-triggerauthentication.yaml.j2`

**Standup wiring:**
- `step_03` (workload monitoring) applies the `TriggerAuthentication` once per namespace (bearer-secret mode only), then the `ScaledObjects` template.
- `step_09` (deploy modelservice) re-applies the `ScaledObjects` template per stack (idempotent).
- Neither step is gated on `is_openshift`.

#### `keda.prometheus` — shared Prometheus connection

| Field | Default | Description |
|-------|---------|-------------|
| `keda.prometheus.baseUrl` | `http://prometheus` | Base URL of the Prometheus instance (no trailing port) |
| `keda.prometheus.port` | `9090` | Prometheus port; assembled with `baseUrl` as `baseUrl:port` |
| `keda.prometheus.authMode` | `none` | Auth mode: `none` or `bearer-secret` |
| `keda.prometheus.secretName` | `""` | Name of a pre-existing Secret in the **deploy namespace** containing `bearerToken` and `ca.crt` keys (`bearer-secret` only) |
| `keda.prometheus.unsafeSsl` | `false` | Skip TLS verification; also omits the `ca` secretTargetRef entry from the TriggerAuthentication |

**Auth modes:**

| `authMode` | TriggerAuthentication created? | Secret required? |
|------------|-------------------------------|-----------------|
| `none` | No | No |
| `bearer-secret` | Yes (`keda-prometheus-auth`) | Yes — user must pre-create it in the deploy namespace |

> **Note:** KEDA's `secretTargetRef` does not support cross-namespace Secret references. The Secret must be in the same namespace as the ScaledObject.

#### `keda.scaledObjects` — list of ScaledObjects to create

Each entry in the list produces one `ScaledObject`. The list is empty by default (no ScaledObjects rendered).

| Field | Default | Description |
|-------|---------|-------------|
| `name` | _(required)_ | `metadata.name` of the ScaledObject |
| `scaleTargetRef.kind` | `Deployment` | Kind of the scale target |
| `scaleTargetRef.name` | `model_id_label + "-decode"` | Name of the scale target; defaults to the model's decode Deployment |
| `minReplicas` | `1` | Minimum replica count |
| `maxReplicas` | `10` | Maximum replica count |
| `pollingInterval` | _(omitted)_ | Seconds between KEDA polls; omit to use KEDA's default (15 s) |
| `triggers` | `[]` | List of KEDA trigger objects (see below) |
| `behavior` | _(omitted)_ | Optional HPA behavior block rendered under `spec.advanced.horizontalPodAutoscalerConfig.behavior` |

Each entry in `triggers` is a raw KEDA trigger. For Prometheus triggers, `serverAddress` is injected automatically from `keda.prometheus`; you do not set it manually.

| Trigger field | Default | Description |
|---------------|---------|-------------|
| `type` | _(required)_ | KEDA trigger type, e.g. `prometheus` |
| `name` | _(omitted)_ | Optional trigger name |
| `metricType` | `AverageValue` | HPA metric type |
| `query` | _(required)_ | PromQL query string |
| `threshold` | `"1"` | Scale-up threshold |
| `activationThreshold` | `"0"` | Activation threshold (KEDA `activationThreshold`) |

#### Example

```yaml
keda:
  prometheus:
    baseUrl: http://prometheus-operated.monitoring.svc.cluster.local
    port: 9090
    authMode: none

  scaledObjects:
    - name: decode-saturation
      scaleTargetRef:
        kind: Deployment
        name: ""              # defaults to model_id_label + "-decode"
      minReplicas: 1
      maxReplicas: 10
      pollingInterval: 15
      triggers:
        - type: prometheus
          name: kv-cache
          metricType: AverageValue
          query: |
            max(inference_pool_average_kv_cache_utilization{namespace="my-ns"})
          threshold: "0.7"
          activationThreshold: "0"
```

For `bearer-secret` auth, first create a Secret in the deploy namespace:

```bash
kubectl create secret generic prometheus-bearer \
  --from-literal=bearerToken="<token>" \
  --from-file=ca.crt=/path/to/ca.crt \
  -n <deploy-namespace>
```

Then set:

```yaml
keda:
  prometheus:
    authMode: bearer-secret
    secretName: prometheus-bearer
```

---

## Container Images

The tool uses several container images across different components. Which config key controls which image depends on the deployment method (standalone vs. modelservice).

### Image Config Paths

All images are defined in `defaults.yaml`. There are two groups: the shared `images` section and per-component overrides.

**Shared images** (under `images`):

| Key | Default | Used by |
|-----|---------|---------|
| `images.vllm` | `docker.io/vllm/vllm-openai:auto` | Model-server pods whose command launches vLLM; also the fallback for an engine with no entry |
| `images.sglang` | `docker.io/lmsysorg/sglang:<pin>` | Model-server pods whose command launches SGLang |
| `images.trtllm` | `nvcr.io/nvidia/tensorrt-llm/release:<pin>` | Model-server pods whose command launches TensorRT-LLM |
| `images.llmdInferenceSim` | `ghcr.io/llm-d/llm-d-inference-sim:auto` | Model-server pods running the simulator |
| `images.benchmark` | `ghcr.io/llm-d/llm-d-benchmark:auto` | Download job, harness pod, data access pod |
| `images.routerEndpointPicker` | `ghcr.io/llm-d/llm-d-router-endpoint-picker-dev:auto` | llm-d-router EPP |
| `images.routingSidecar` | `ghcr.io/llm-d/llm-d-routing-sidecar:auto` | Modelservice routing sidecar (proxy in front of vLLM) |
| `images.udsTokenizer` | `ghcr.io/llm-d/llm-d-uds-tokenizer:auto` | UDS tokenizer. **Not wired to `router.tokenizer`** -- the llm-d-router chart runs its own `vllm-render` sidecar, configured via `router.tokenizer.image`. Retained only for scenarios that reference it as an init container via `imageKey: udsTokenizer`. |
| `images.python` | `python:3.10` | Utility containers |

**Per-component images** (override the shared defaults):

| Key | Default | Used by |
|-----|---------|---------|
| `<role>.engine.image` | _(falls back to `images.<engine>`)_ | The model-server container of that role (`decode`, `prefill`, `standalone`, `nok8s`) |
| `standalone.launcher.image` | _(falls back to `standalone.engine.image`)_ | Standalone launcher container (repo/tag only) |
| `wva.image` | `ghcr.io/llm-d/llm-d-workload-variant-autoscaler:auto` | Workload Variant Autoscaler |

Each image key has `repository`, `tag`, and `pullPolicy` sub-fields. The one exception is `standalone.launcher` --its pull policy is set via a separate flat key `standalone.launcher.imagePullPolicy` (defaults to `Always`), not nested under `image`.

### Which Template Uses Which Image

| Template | Image Config | Component |
|----------|-------------|-----------|
| `04_download_job.yaml.j2` | `images.benchmark` | Model download job |
| `06_pod_access_to_harness_data.yaml.j2` | `images.benchmark` | Harness data access pod |
| `12_router-values.yaml.j2` | `images.routerEndpointPicker` | llm-d-router EPP |
| `13_ms-values.yaml.j2` (decode) | `images.vllm` | Decode pods in modelservice |
| `13_ms-values.yaml.j2` (prefill) | `images.vllm` | Prefill pods in modelservice |
| `13_ms-values.yaml.j2` (sidecar) | `images.routingSidecar` | Routing sidecar in modelservice |
| `13_ms-values.yaml.j2` (init containers) | `images.<imageKey>` | Per-init-container, via `imageKey:` (defaults to `images.benchmark`) |
| `14_standalone-deployment_yaml.j2` | `standalone.engine.image` | Standalone model-server container |
| `14_standalone-deployment_yaml.j2` (launcher) | `standalone.launcher.image` | Standalone launcher container |
| `19_wva-kustomize.yaml.j2` | `wva.image` | Workload Variant Autoscaler |
| `20_harness_pod.yaml.j2` | `images.benchmark` | Benchmark harness pod |

### Fallback Chains

Templates use Jinja2 `default()` filters to create fallback chains. If a per-component image isn't set, the template falls back to the shared `images` section.

**Model-server container of any role:**

```
<role>.engine.image.repository  maps to  images.<engine>.repository
<role>.engine.image.tag         maps to  images.<engine>.tag
<role>.engine.image.pullPolicy  maps to  images.<engine>.pullPolicy
```

This one is filled in by the engine resolver rather than a Jinja `default()`
chain, and it is per sub-field: a role that pins only `repository` still gets
its `tag` and `pullPolicy` from `images.<engine>`. Which `images` entry that is
follows from the engine the role's command launches (`vllm serve` reads
`images.vllm`, `python3 -m sglang.launch_server` reads `images.sglang`, and so
on), so a scenario that switches engines does not also have to restate the
image. An `auto` tag is resolved at render time by the `VersionResolver`, on
both the shared `images.<engine>.tag` and a per-role override.

**Standalone launcher container** (three-level chain for repo/tag):

```
standalone.launcher.image.repository  maps to  standalone.engine.image.repository  maps to  images.vllm.repository
standalone.launcher.image.tag         maps to  standalone.engine.image.tag         maps to  images.vllm.tag
standalone.launcher.imagePullPolicy   maps to  'Always' (hardcoded default, no fallback chain)
```

Note: the launcher's `imagePullPolicy` is a flat key on `standalone.launcher`, not nested under `standalone.launcher.image`. It does not inherit from `standalone.engine.image.pullPolicy`.

**Modelservice decode/prefill containers:**

```
decode container imagePullPolicy:  decode.engine.image.pullPolicy  maps to  images.<engine>.pullPolicy  maps to  'IfNotPresent'
prefill container imagePullPolicy: prefill.engine.image.pullPolicy maps to  images.<engine>.pullPolicy  maps to  'IfNotPresent'
```

Setting `images.vllm.pullPolicy: Always` in your scenario applies to both decode and prefill containers that run vLLM. Per-role overrides via `decode.engine.image.pullPolicy` or `prefill.engine.image.pullPolicy` take precedence.

**Everything else** (download job, harness, etc.) reads directly from the `images` section with no fallback chain.

### Overriding Images

**One role only** (a standalone pod, or a single decode/prefill role):

Override that role's `engine.image`:

```yaml
scenario:
  - name: "my-standalone"
    standalone:
      enabled: true
      engine:
        image:
          repository: docker.io/vllm/vllm-openai
          tag: v0.8.5
          pullPolicy: Always
```

**Modelservice deployment** (optimized-baseline, pd-disaggregation, etc.):

Override `images.vllm` in your scenario. The `pullPolicy` applies to both decode and prefill containers:

```yaml
scenario:
  - name: "my-modelservice"
    images:
      vllm:
        repository: quay.io/myorg/vllm-dev
        tag: latest
        pullPolicy: Always
```

To override pull policy for a specific role only:

```yaml
scenario:
  - name: "my-modelservice"
    images:
      vllm:
        repository: quay.io/myorg/vllm-dev
        tag: latest
    decode:
      vllm:
        imagePullPolicy: Always    # decode only
```

**Benchmark harness / download job:**

Override `images.benchmark`:

```yaml
scenario:
  - name: "my-deployment"
    images:
      benchmark:
        repository: my-registry/llm-d-benchmark
        tag: dev-branch
```

**Router EPP:**

Override `images.routerEndpointPicker`:

```yaml
scenario:
  - name: "my-deployment"
    images:
      routerEndpointPicker:
        repository: my-registry/llm-d-router-endpoint-picker
        tag: v1.2.3
```

**Routing sidecar:**

This is read directly from `images.routingSidecar` -- override the same way:

```yaml
scenario:
  - name: "my-deployment"
    images:
      routingSidecar:
        tag: v0.8.0
```

There is no per-block image field on `routing.proxy` -- the `images.*` entry is the single source of truth.

**EPP tokenizer sidecar:**

`router.tokenizer` is the exception: it is passed through to the llm-d-router
chart verbatim, so its image is set on the block itself rather than under
`images.*`. Useful when the chart default (`docker.io/vllm/vllm-openai-cpu`,
amd64-only) does not match your nodes:

```yaml
scenario:
  - name: "my-deployment"
    modelservice:
      router:
        tokenizer:
          enabled: true
          image:
            registry: my-registry
            repository: vllm-openai-cpu
            tag: v0.19.1
```

**Init containers** (`decode.initContainers[*]`, `prefill.initContainers[*]`, `standalone.initContainers[*]`):

Three options, in order of preference:

1. `imageKey: <entry>` -- references `images.<entry>` from `defaults.yaml` (recommended; tracks version bumps automatically). Inherits `imagePullPolicy` from the same entry when not set on the init container.

2. `image: <full-string>` -- direct override; supports `:auto` tag resolution via the registry. Use for one-off images that don't have an `images.*` entry.

3. Neither -- the template falls back to `images.benchmark`.

```yaml
decode:
  initContainers:
    - name: preprocess
      imageKey: benchmark              # tracks images.benchmark
      command: [...]
    - name: my-custom
      image: my-registry/init:v1.0     # one-off override; no images.* entry needed
      command: [...]
```

To change the image used by every `imageKey: benchmark` reference at once, override `images.benchmark` in your scenario (same as any other entry above). Setting both `image:` and `imageKey:` on the same init container is a config error and aborts plan generation; an unknown `imageKey` does too (the error message lists the available keys).

**How to tell which one to use:** override `images.<engine>` to move every role that runs that engine, and `<role>.engine.image` to move one role only. The per-role key is the same on `decode`, `prefill`, `standalone` and `nok8s`, so which deployment method the scenario uses does not change where the override goes. You can verify by running `plan` and inspecting the rendered YAML in the stack output directory.

**Image override logging:** When a scenario pins an image to a non-auto tag, the renderer logs the override during plan rendering. For example: `Image override: vllm pinned to us.icr.io/...:v1.1.1`. This makes it easy to see which images differ from the auto-resolved defaults.

After standup, the deployed images are recorded in the `llm-d-benchmark-standup-parameters` ConfigMap:

```bash
oc get configmap llm-d-benchmark-standup-parameters -n <namespace> -o yaml
```

---

## Private Registries (`engine.pullSecret`)

Set `engine.pullSecret` to the name of an existing pull secret and
`imagePullSecrets` is added to every pod spec the benchmark renders:

```yaml
engine:
  pullSecret: secret-example  # pragma: allowlist secret
```

Or without editing the scenario:

```bash
llmdbenchmark --spec examples/spyre standup --set 'engine.pullSecret=secret-example'  # pragma: allowlist secret
```

The secret must already exist in the namespace -- nothing here creates it:

```bash
oc create secret docker-registry secret-example \
  --docker-server=<registry-host> \
  --docker-username=<username> \
  --docker-password="$REGISTRY_PASSWORD" \
  -n <namespace>
```

Covered pod specs: decode and prefill (via the modelservice chart's pod-level
`extraConfig`), standalone, the model download job and daemonset, the harness
pod, the harness data-access pod, and the ephemeral pods used by smoketests.

> [!IMPORTANT]
> This does **not** reach the EPP pod or its `vllm-render` tokenizer sidecar.
> The llm-d-router chart exposes no `imagePullSecrets` field, so a private
> `router.tokenizer.image` (or EPP image) needs a different mechanism -- link
> the secret to the router's service account, or use the cluster-wide pull
> secret:
> ```bash
> oc secrets link <model-id-label> secret-example --for=pull -n <namespace>
> ```

---

## Pod Scheduling

### Priority Class

Set `priorityClassName` to control pod scheduling priority. This maps to the Kubernetes `priorityClassName` field on the pod spec.

**Set for all pods (recommended):**

```yaml
engine:
  priorityClassName: "high-priority"
```

This applies to decode, prefill, and standalone pods. Matches the bash `LLMDBENCH_VLLM_COMMON_PRIORITY_CLASS_NAME`.

**Override per role:**

```yaml
decode:
  priorityClassName: "high-priority"
prefill:
  priorityClassName: "low-priority"
```

Per-role values override `engine.priorityClassName`.

**Disable (default):**

Leave empty or set to `"none"`. No `priorityClassName` is rendered and pods use the cluster default priority.

> [!NOTE]
> The PriorityClass must already exist on the cluster. Create it with `kubectl apply` before standup. Example: `kubectl create priorityclass high-priority --value=1000 --global-default=false`

### Scheduler Name

Override the pod scheduler (e.g., for Spyre which requires `spyre-scheduler`):

```yaml
schedulerName: spyre-scheduler
```

This sets `schedulerName` on all modelservice pods. If not set, Kubernetes uses the default scheduler.

---

## Scenarios

Scenario files provide deployment-specific overrides that are merged on top of `defaults.yaml`. They configure things like model name, GPU count, namespace, image tags, and deployment topology.

### `scenarios/guides/`

Map directly to the [llm-d well-lit-path guides](https://github.com/llm-d/llm-d/tree/main/guides). Each scenario reproduces the deployment described in its corresponding guide.

| Scenario | Description |
|----------|-------------|
| `agentic-serving.yaml` | Reasoning and tool-call serving driven by OTel-trace replay |
| `epp-keda-saturation.yaml` | Optimized baseline plus EPP+KEDA saturation autoscaling (no WVA) |
| `fast-model-actuation-base.yaml` | Fast model actuation, kustomize deploy only |
| `fast-model-actuation-keda.yaml` | Fast model actuation plus scale-from-zero KEDA |
| `flow-control.yaml` | Optimized baseline with custom EPP flow-control plugins |
| `multimodal-serving-aggregation.yaml` | Multimodal serving, aggregated prefill and decode |
| `multimodal-serving-e-disaggregation.yaml` | Multimodal serving, encode disaggregated from prefill/decode |
| `nok8s.yaml` | vLLM, EPP and Envoy as local containers, no cluster |
| `optimized-baseline.yaml` | Qwen3-32B with inference scheduling plugins |
| `p2p-kv-cache-sharing.yaml` | Peer-to-peer KV cache sharing between pods |
| `pd-disaggregation.yaml` | Prefill/decode disaggregation |
| `precise-prefix-cache-routing.yaml` | Prefix cache aware routing |
| `predicted-latency-routing.yaml` | Routing on predicted request latency |
| `tiered-prefix-cache.yaml` | Tiered CPU/GPU prefix cache |
| `wide-ep.yaml` | Wide expert parallelism (DisaggregatedSet) |
| `workload-autoscaling.yaml` | Optimized baseline plus the Workload Variant Autoscaler |

#### Kustomize Deployment (`-t kustomize`)

Guide scenarios can be deployed via kustomize instead of the default modelservice/standalone methods. The tool parses the guide's README.md to extract `helm` and `kubectl` commands, resolves variables, and executes them in order.

```bash
llmdbenchmark --spec guides/optimized-baseline standup \
    -t kustomize -p my-namespace \
    --llmd-repo-path /path/to/llm-d
```

##### Kustomize Config Fields

```yaml
kustomize:
  enabled: true
  guideName: "optimized-baseline"    # Guide directory name under guides/
  repoPath: ""                       # Local path to llm-d repo (auto-cloned if empty)
  repoRef: "main"                    # Git ref to checkout
  gaieVersion: ""                    # GAIE CRD bundle version (auto-detected from README if empty)
  routerChartVersion: ""             # llm-d-router chart version (auto-detected from README; defaults to v0)
  acceleratorBackend: "gpu/vllm"     # Modelserver backend path
  monitoring: false                  # Apply monitoring kustomize overlay
  overlayPath: ""                    # Path to additional kustomize overlay directory
  extraHelmValues: []                # Extra -f args for helm install
  extraHelmSets: {}                  # Extra --set args for helm install
  guideVariableOverrides: {}         # Override/fill the guide README's ${VAR} values
  deployTimeout: 900                 # Seconds to wait for pods
  patches: []                        # Inline strategic merge patches (see below)
```

##### Inline Patches

Use `patches` to apply kustomize strategic merge patches on top of the guide's modelserver base. Each entry is a YAML manifest that gets merged with the matching resource.

Set replica counts (important -- upstream guides may set high defaults):

```yaml
kustomize:
  patches:
    - patch: |
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: decode
        spec:
          replicas: 2
    - patch: |
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: prefill
        spec:
          replicas: 1
```

Add a volume mount:

```yaml
kustomize:
  patches:
    - patch: |
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: decode
        spec:
          template:
            spec:
              volumes:
                - name: triton-cache
                  emptyDir: {}
              containers:
                - name: modelserver
                  volumeMounts:
                    - mountPath: /.triton
                      name: triton-cache
```

Inject an HF_TOKEN env var for gated models (e.g. Llama). Set `HF_TOKEN` in your shell environment before running -- the tool auto-creates a `llm-d-hf-token` Kubernetes secret in the namespace. Then add this patch so the pod reads it:

```yaml
kustomize:
  patches:
    - patch: |
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: decode
        spec:
          template:
            spec:
              containers:
                - name: modelserver
                  env:
                    - name: HF_TOKEN
                      valueFrom:
                        secretKeyRef:
                          name: llm-d-hf-token
                          key: HF_TOKEN
```

Env vars from patches are merged with the guide's existing env vars, not replaced. The container is matched by `name: modelserver`.

Set resource requests:

```yaml
kustomize:
  patches:
    - patch: |
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: decode
        spec:
          template:
            spec:
              containers:
                - name: modelserver
                  resources:
                    limits:
                      nvidia.com/gpu: "4"
                    requests:
                      nvidia.com/gpu: "4"
```

Multiple patches can be combined in a single list -- they are applied in order.

##### Environment Variables

| Variable | Effect |
|----------|--------|
| `HF_TOKEN` | Auto-creates a `llm-d-hf-token` secret in the namespace |
| `LLMDBENCH_PRIORITY_CLASS` | Injects `priorityClassName` on the decode Deployment (for clusters with pod priority/preemption policies) |

### `scenarios/examples/`

Minimal starting points for common hardware:

| Scenario | Description |
|----------|-------------|
| `cpu.yaml` | CPU-only deployment (no GPU, uses vllm-cpu-release image) |
| `gpu.yaml` | Standard NVIDIA GPU deployment |
| `engines.yaml` | One small stack, three engines: vLLM live, SGLang and TensorRT-LLM commented out beneath it |
| `sim.yaml` | Simulated inference (llm-d-inference-sim, no GPU required; minimal PVC and resources) |
| `spyre.yaml` | IBM Spyre accelerator |

`engines.yaml` is the one to read for engine-agnosticism: the three engines differ
only in `decode.engine.command` (plus, for TensorRT-LLM, three keys commented in
place beside the ones they belong to). No engine-specific configuration key
exists, so a new engine needs no new scenario shape -- and no new scenario file.
`--engine sglang` / `--engine trtllm` switch the commented block in without
editing anything; see [Engine Command](#engine-command).

### `scenarios/cicd/`

Used by automated CI/CD pipelines:

| Scenario | Description |
|----------|-------------|
| `kind.yaml` | Kind cluster with `llm-d-inference-sim` (no GPU, CPU-only, public model). Exercises the full modelservice and standalone paths in CI. |
| `gke.yaml` | Google Kubernetes Engine with H100 |
| `cks.yaml` | Cloud Kubernetes Service with H200 |
| `ocp.yaml` | OpenShift Container Platform with Istio |

### Creating a New Scenario

1. Start from an existing scenario or from scratch
2. Only specify the values you want to override -- everything else comes from `defaults.yaml`
3. Place it under `config/scenarios/` in the appropriate subdirectory

Example minimal scenario:

```yaml
scenario:
  - name: "my-deployment"

    model:
      name: meta-llama/Llama-3.1-8B
      path: models/meta-llama/Llama-3.1-8B
      huggingfaceId: meta-llama/Llama-3.1-8B

    # Deployment method -- choose one
    modelservice:
      enabled: true
    standalone:
      enabled: false

    decode:
      replicas: 2
      resources:
        limits:
          memory: 64Gi
          cpu: "16"
        requests:
          memory: 64Gi
          cpu: "16"

    harness:
      name: inference-perf
      experimentProfile: sanity_random.yaml

    workDir: "~/data/my-deployment"
```

---

## Specifications

Specification files are the entry points for the CLI. Each is a Jinja2 template (`.yaml.j2`) that declares paths to the defaults, templates, and scenario files, plus optional experiment definitions.

### Required Fields

Every specification must declare three paths:

```yaml
{% set base_dir = base_dir | default('../') -%}
base_dir: {{ base_dir }}

values_file:
  path: {{ base_dir }}/config/templates/values/defaults.yaml

template_dir:
  path: {{ base_dir }}/config/templates/jinja
```

### Optional Fields

```yaml
scenario_file:
  path: {{ base_dir }}/config/scenarios/guides/optimized-baseline.yaml

experiments:
  - name: "experiment-name"
    attributes:
      - name: "setup"
        factors: [...]
        treatments: [...]
      - name: "run"
        factors: [...]
        treatments: [...]
```

### Specification Auto-Discovery

The `--spec` flag supports three input forms --you don't need to type the full path:

| Form | Example | Resolves to |
|------|---------|-------------|
| **Bare name** | `--spec gpu` | `config/specification/examples/gpu.yaml.j2` |
| **Category/name** | `--spec guides/optimized-baseline` | `config/specification/guides/optimized-baseline.yaml.j2` |
| **Full path** | `--spec config/specification/guides/optimized-baseline.yaml.j2` | Used as-is |

The `.yaml.j2` suffix is added automatically. If a bare name matches files in multiple categories, you'll be prompted to disambiguate with the category prefix.

### The `base_dir` Variable

All paths are relative to `base_dir`, which defaults to `../` (the repository root when running from the repo directory). Override it with `--bd`:

```bash
llmdbenchmark --bd /path/to/repo --spec guides/optimized-baseline plan
```

### Creating a New Specification

1. Create a scenario YAML under `config/scenarios/` with your deployment overrides
2. Create a specification template under `config/specification/` in the appropriate category subdirectory:

```yaml
{% set base_dir = base_dir | default('../') -%}
base_dir: {{ base_dir }}

values_file:
  path: {{ base_dir }}/config/templates/values/defaults.yaml

template_dir:
  path: {{ base_dir }}/config/templates/jinja

scenario_file:
  path: {{ base_dir }}/config/scenarios/my-scenario.yaml
```

3. Run: `llmdbenchmark --spec my-spec plan`

#### Naming and Collisions

Choose a **unique file name** for your specification. Auto-discovery searches across all subdirectories under `config/specification/`, so two files with the same base name in different categories will collide:

```text
config/specification/
    guides/optimized-baseline.yaml.j2     <- exists
    examples/optimized-baseline.yaml.j2   <- collision!
```

Running `--spec optimized-baseline` with both present produces an error:

```text
Ambiguous specification name 'optimized-baseline' matches 2 files:
  - /path/to/config/specification/examples/optimized-baseline.yaml.j2
  - /path/to/config/specification/guides/optimized-baseline.yaml.j2

Use category/name to disambiguate, e.g.
'--spec guides/optimized-baseline' or '--spec examples/optimized-baseline'.
```

To avoid this:

- **Use a distinct name** that reflects your use case (e.g. `my-team-inference.yaml.j2` instead of reusing `optimized-baseline.yaml.j2`)
- **Or always use category/name** when specs share a base name: `--spec guides/optimized-baseline`

### Experiments

To add parameter sweeps, include an `experiments` section. Experiments have two attribute categories:

- **`setup`** -- Parameters that change the deployment (e.g., replicas, scheduler plugin). Each treatment generates a separate rendered stack.
- **`run`** -- Parameters that change the benchmark workload (e.g., concurrency, prompt length). Used during the run phase, not standup.

Each category contains:

| Field | Purpose |
|-------|---------|
| `factors` | Parameters being varied, each with a list of `levels` (possible values) |
| `constants` | Fixed parameters applied to every treatment (optional) |
| `treatments` | Explicit combinations of factor levels to test |

### Available Specifications

**Guides:**

Every guide specification stands up on its own. Sweeps are supplied
separately with `--experiments` (see [Experiments](#experiments)); the
right column names the file under `experiments/` that matches the guide.

| Specification | Matching experiment |
|---------------|---------------------|
| `agentic-serving.yaml.j2` | -- |
| `epp-keda-saturation.yaml.j2` | -- |
| `fast-model-actuation-base.yaml.j2` | -- |
| `fast-model-actuation-keda.yaml.j2` | -- |
| `flow-control.yaml.j2` | -- |
| `multimodal-serving-aggregation.yaml.j2` | -- |
| `multimodal-serving-e-disaggregation.yaml.j2` | -- |
| `nok8s.yaml.j2` | -- |
| `optimized-baseline.yaml.j2` | `optimized-baseline.yaml` |
| `p2p-kv-cache-sharing.yaml.j2` | -- |
| `pd-disaggregation.yaml.j2` | `pd-disaggregation.yaml` |
| `precise-prefix-cache-routing.yaml.j2` | `precise-prefix-cache-aware.yaml` |
| `predicted-latency-routing.yaml.j2` | -- |
| `tiered-prefix-cache.yaml.j2` | `tiered-prefix-cache.yaml` |
| `wide-ep.yaml.j2` | -- |
| `workload-autoscaling.yaml.j2` | -- |

**Examples:** `cpu.yaml.j2`, `engines.yaml.j2`, `eval-containers-aider-polyglot.yaml.j2`, `eval-containers-aider-polyglot-gpu.yaml.j2`, `eval-containers-gaia.yaml.j2`, `eval-containers-gaia-gpu.yaml.j2`, `fma.yaml.j2`, `gpu.yaml.j2`, `intel-xpu.yaml.j2`, `launcher.yaml.j2`, `multi-model-optimized-baseline.yaml.j2`, `sim.yaml.j2`, `spyre.yaml.j2`, `spyre-s390x.yaml.j2`

**CI/CD:** `cks.yaml.j2`, `gke.yaml.j2`, `kind.yaml.j2`, `ocp.yaml.j2`, `ocp-keda.yaml.j2`, `ocp-keda-fma-hotstart.yaml.j2`, `ocp-keda-fma-warmstart.yaml.j2`

**Experimental:** `kimi-k3-h100.yaml.j2`

---

## Usage

```bash
# Plan (render templates into manifests)
llmdbenchmark --spec guides/optimized-baseline plan

# Standup (plan + apply to cluster)
llmdbenchmark --spec guides/optimized-baseline standup

# Dry run
llmdbenchmark --spec guides/optimized-baseline --dry-run standup

# Teardown
llmdbenchmark --spec guides/optimized-baseline teardown

# Override namespace at runtime
llmdbenchmark --spec guides/optimized-baseline standup -p my-ns

# Override deployment method
llmdbenchmark --spec guides/optimized-baseline standup -t standalone

# Use category/name to disambiguate
llmdbenchmark --spec guides/optimized-baseline standup

# Full path still works
llmdbenchmark --spec config/specification/guides/optimized-baseline.yaml.j2 standup
```
