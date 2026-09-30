## Concept
`llm-d-benchmark` provides its own automated framework for the standup of stacks serving large language models in a Kubernetes cluster.

## Motivation
In order to allow reproducible and flexible experiments, and taking into account that the configuration paramaters have significant impact on the overall performance, it is necessary to provide the user with the ability to `standup` and `teardown` stacks.

## Methods
Currently, the following standup methods are supported
a) "Standalone", with multiple VLLM `pods` controlled by a `deployment` behind a single `service`
b) "llm-d", which leverages a combination of [llm-d-infra](https://github.com/llm-d-incubation/llm-d-infra.git) and [llm-d-modelservice](https://github.com/llm-d/llm-d-model-service.git) to deploy a full-fledged `llm-d` stack
c) "No-Kubernetes (`nok8s`)", which runs the routing stack (vLLM + EPP + Envoy) as plain `docker`/`podman` containers on a single host, with **no cluster** -- see [No-Kubernetes deploy method](nok8s.md)

## Scenarios
All the information required for the standup of a stack is contained on a "scenario file". This information is encoded in the form of environment variables, with default values defined in `config/defaults.yaml` which can be then overriden inside a [scenario file](../config/scenarios) (YAML-based) or via [specification templates](../config/specification) (Jinja2 `.yaml.j2` files).

### Multi-Stack Scenarios

A scenario may define more than one stack in its `scenario:` list. Standup
iterates every per-stack step across all stacks (in parallel, bounded by
`--parallel`), so you can stand up N models behind one gateway in a single
`llmdbenchmark standup` invocation. Scenario-wide config (gateway class,
shared HTTPRoute, EPP plugin config, chart versions) lives in an optional
top-level `shared:` block that's merged into every stack before per-stack
overrides.

Cluster-scoped infrastructure that would race with itself across N parallel
standup executions is deduplicated at render time - only the first stack
emits the istio control-plane helmfile and the `infra-llmdbench` Helm
release; subsequent stacks render empty files for those templates. When a
multi-stack scenario does enable WVA, controller installation is
deduplicated at the step level too (one per `wva.namespace`).

Currently shipped multi-stack example:

- [`examples/multi-model-optimized-baseline`](../config/scenarios/examples/multi-model-optimized-baseline.yaml) -
  the [optimized-baseline](../config/scenarios/guides/optimized-baseline.yaml)
  guide deployed twice: two models (Qwen3-0.6B + Meta-Llama-3.1-8B), each
  with its own EPP + InferencePool + decode Deployment, one HTTPRoute with
  two backendRefs routing by path prefix (`/qwen3-06b/*` -> Qwen pool,
  `/llama-31-8b/*` -> Llama pool).

See [`config/README.md`](../config/README.md#method-1-scenario-file-recommended-for-deployment-specific-config)
for the `shared:` merge semantics, the developer guide's
[Multi-Stack Scenarios](developer-guide.md#multi-stack-scenarios-and-the-shared-block)
section for the render-engine details, and
[multi-model.md](multi-model.md) for the day-to-day operations cookbook.

`--stack NAME[,NAME...]` (also `LLMDBENCH_STACK=NAME`) restricts standup to
a subset of rendered stacks - handy for re-deploying a single pool after a
scenario edit without tearing down siblings. Global steps (cluster admin
prereqs, shared-infra helmfile, scenario-wide PVCs) still run as usual;
only per-stack steps (06+ for standup) are filtered. Unknown names fail
loudly with a list of valid ones.

```bash
# One stack:
llmdbenchmark --spec examples/multi-model-optimized-baseline standup -p my-namespace --stack qwen3-06b

# Multiple named stacks (comma-separated):
llmdbenchmark --spec examples/multi-model-optimized-baseline standup -p my-namespace --stack qwen3-06b,llama-31-8b
```

The same flag works on `smoketest`, `run`, and `teardown` with identical
semantics, so you can scope every lifecycle phase to the same subset.

## Overriding scenario values from the CLI (`--set`)

A scenario variant that differs from an existing one in only a handful of
fields does not need its own YAML file. Every subcommand that renders
templates accepts `--set`, which deep-merges dotted-path values on top of
the scenario:

```bash
# Run the SGLang flavour of a guide without a separate scenario file
llmdbenchmark --spec guides/optimized-baseline standup \
  -t kustomize --set kustomize.acceleratorBackend=gpu/sglang
```

Pairs are comma-separated and the flag is repeatable. Values are parsed as
YAML, so `4`, `true`, `[a, b]` and `{x: 1}` mean what they would inside the
scenario file; commas inside `[]`, `{}` or quotes belong to the value.

> [!WARNING]
> **Multi-line values are folded onto one line.** A value containing real
> newlines is read as a YAML plain scalar, so its line breaks collapse into
> spaces -- which silently changes the meaning of a shell command
> (`export FOO=1`⏎`vllm serve` becomes `export FOO=1 vllm serve`). To keep
> the breaks, wrap the value in double quotes so `\n` is an escape:
> `--set 'decode.engine.command="export FOO=1\nvllm serve /model-cache/x"'`.
> For a full multi-line engine command, prefer the scenario file or
> `--cluster-config` -- `--set` is best suited to single-line values.

> [!IMPORTANT]
> `--set` always means the **scenario**, on every subcommand. It is not the
> same as `run`/`experiment`'s `-o/--overrides`, which overrides the
> **workload profile**. Those two are separate flags and can be combined:
> `run --set decode.replicas=4 -o max-concurrency=8`. `standup` has no
> workload profile, so it accepts `--set` only.

The same value can be supplied via `LLMDBENCH_SET`. Pass `--set` to every
lifecycle phase (`plan`/`standup`/`smoketest`/`run`/`teardown`) so each one
renders the same plan -- these phases re-render templates, and a phase that
misses the flag will disagree with what was deployed.

### Scoping overrides in multi-stack scenarios

Prefix the key with a stack name, or an fnmatch glob, to scope an override
in a [multi-stack scenario](#multi-stack-scenarios). Unprefixed applies to
every stack:

```bash
# every stack
llmdbenchmark --spec examples/multi-model-optimized-baseline standup --set decode.replicas=2

# one stack; both are still deployed
llmdbenchmark --spec examples/multi-model-optimized-baseline standup \
  --set 'qwen3-06b:decode.replicas=4,llama-31-8b:decode.replicas=1'

# a common floor with one exception
llmdbenchmark --spec examples/multi-model-optimized-baseline standup \
  --set 'decode.resources.limits.memory=64Gi' \
  --set 'llama-31-8b:decode.resources.limits.memory=32Gi'

# every stack whose name ends in -8b
llmdbenchmark --spec examples/multi-model-optimized-baseline standup \
  --set '*-8b:decode.resources.limits.memory=64Gi'
```

When several selectors match a stack they are applied by specificity --
global, then globs, then exact names -- so the exception above wins
regardless of the order the flags were typed. A selector that matches no
stack in the scenario is a hard error, not a silent no-op.

`--stack` and override selectors are orthogonal: `--stack` chooses which
stacks are **deployed**, a selector chooses which stacks are **modified**.
Note that an unprefixed `--set` applies to every stack even when `--stack`
narrows the deployment, which differs from `-m/--models` (that one scopes
itself to a single filtered stack).

### Precedence and limits

Highest wins:

```
defaults.yaml → shared: → stack block → --cluster-config → --set
  → DoE setup.treatments → dedicated flags (-m, -t, --gateway-class,
                                            --monitoring, --wva)
```

`--set` beats the stack's own block -- unlike a value in `shared:`, which
loses to it. DoE `setup.treatments` beat `--set`, because the treatment is
the deliberate sweep factor.

The dedicated flags sit at the top because they are applied by resolver
functions that run *after* the whole merge, not as another merge layer. So
`-m facebook/opt-125m` wins over both `--set model.name=...` and a
treatment that sets `model.name`. Use `--set` for keys with no dedicated
flag; when a flag exists, the flag is authoritative.

Every applied override is logged with its previous value
(`[stack] Scenario override: decode.replicas: 1 -> 4`), and an override
whose *parent* path does not exist warns about a possible typo.

**Lists are assigned whole, never indexed.** A dotted path cannot address a
list element, so `--set engine.volumeMounts.0.mountPath=/x` is rejected
rather than silently replacing the whole list. Assign the list instead:

```bash
--set 'engine.volumeMounts=[{name: dshm, mountPath: /dev/shm}]'
```

(This differs from `run -o`, which overrides the workload profile and *does*
support list indices.)

Three things overrides cannot do:

- **Add or remove a stack.** Scenarios differing in stack *count* cannot be
  collapsed into one file.
- **Change a stack's `name`.** It names the plan output directory and is
  read before the merge.
- **Move the workspace via `workDir`.** That is read before rendering; use
  `--workspace` instead.

## Multiple steps
The full standup of a stack is a multi-step process. The [lifecycle](lifecycle.md) document go into more details explaning the meaning of each different individual step.

## Recovering from crashed pods (`--pod-restart-budget`)

Some pods come up broken and only recover once deleted -- the replacement the
controller creates is healthy. By default a pod that lands in a crash state
fails the readiness wait immediately, and with it the whole standup.

`--pod-restart-budget N` lets standup absorb that:

```bash
llmdbenchmark standup --spec <spec> --pod-restart-budget 3
```

```bash
LLMDBENCH_POD_RESTART_BUDGET=3 llmdbenchmark standup --spec <spec>
```

The budget is a **single total for the whole standup** -- shared by every pod,
every readiness wait, and every stack, not a per-pod allowance. `3` means at
most three pod deletions in total, whether they all hit one stubborn decode
pod or one each across three stacks.

Only failures a restart can plausibly fix are retried:

| Situation | Behavior |
|---|---|
| `CrashLoopBackOff`, `Error`, `OOMKilled`, pod phase `Failed` | Deleted and retried while budget remains |
| `ImagePullBackOff`, `ErrImagePull`, `InvalidImageName`, `CreateContainerConfigError` | Fails immediately -- an identical replacement fails identically |
| Pod with no controller to recreate it | Fails immediately -- deleting it means it never comes back |
| Pod already `Terminating` | Left alone; never charged twice |
| Budget exhausted | Fails with `Pod restart budget exhausted (N/N)` |

Each restart adds `--pod-restart-grace` seconds (default `300`) to that wait's
deadline, since the replacement pod re-pulls its image and reloads the model
from scratch. Without it, a crash late in a wait would leave the replacement
no time to become Ready.

Before a pod is deleted, its `describe` output, current logs,
previous-container logs, and events are written to
`<workspace>/setup/logs/pod-restarts/`. Every consumed restart is reported at
the end of standup, so a run that only converged after deleting pods does not
look identical to one that came up clean.

Default is `0` (disabled), which behaves exactly as standup always has.

> [!TIP]
> If you find yourself needing a large budget, the pods are probably failing
> for a structural reason. Check the captured diagnostics before raising it --
> the previous-container logs are usually the ones that explain a crash loop.

## Use
A scenario is a YAML file you write by hand. Once written, it is what `llmdbenchmark standup`, `llmdbenchmark run` and `llmdbenchmark teardown` operate on: it names the stack, states each role's engine launch command, and says everything Kubernetes needs to put around it.

> [!NOTE]
> `llmdbenchmark experiment` is a command that **combines** `llmdbenchmark standup`, `llmdbenchmark run` and `llmdbenchmark teardown` into a single operation. Therefore, the command line parameters supported by the former is a combination of the latter three.

### What comes from the command line

A few things are properties of *this invocation* rather than of the deployment, so they are passed per run instead of written in the file. Each has an environment-variable form: `LLMDBENCH_` plus the flag's long name, upper-cased.

| Flag                            | Environment variable      | Meaning                                                                                                                                                     |
| ------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--spec`/`--specification_file` | `LLMDBENCH_SPEC`          | The scenario file. A path without a leading directory resolves inside `config/scenarios`, so `--spec guides/optimized-baseline` is enough.                   |
| `--stack`                       | `LLMDBENCH_STACK`         | Which named stacks of a multi-stack file to act on (see [Multi-Stack Scenarios](#multi-stack-scenarios)). Omit it and every stack in the file is deployed.   |
| `-p`/`--namespace`              | `LLMDBENCH_NAMESPACE`     | Namespace to stand the stack up in; overrides `namespace.name`.                                                                                              |
| `-m`/`--models`                 | `LLMDBENCH_MODELS`        | Comma-separated model list to stand up, one stack per model; overrides `model.name`.                                                                         |
| `-t`/`--methods`                | `LLMDBENCH_METHODS`       | Comma-separated standup methods: `modelservice`, `standalone`, `kustomize`, `nok8s`, `fma`.                                                                  |
| `--gateway-class`               | `LLMDBENCH_GATEWAY_CLASS` | Router topology for this run; overrides `gateway.className`.                                                                                                |
| `--set`                         | `LLMDBENCH_SET`           | Override any scenario key inline (see [Overriding scenario values from the CLI](#overriding-scenario-values-from-the-cli---set)).                            |
| `--kubeconfig`                  | `LLMDBENCH_KUBECONFIG`    | Kubeconfig to use. Without it, the current context is used.                                                                                                 |
| `--non-admin`                   | `LLMDBENCH_NON_ADMIN`     | Skip the steps that require cluster-admin (CRDs, gateway provider install).                                                                                 |
| `--dry-run`                     | `LLMDBENCH_DRY_RUN`       | Render the plan under `<workspace>/<run-id>/plan/` and stop before changing the cluster.                                                                     |

`llmdbenchmark <subcommand> --help` is the authoritative list; the same `LLMDBENCH_`-prefixed rule applies to every flag it shows.

Hugging Face credentials are read from the environment, never from the scenario, so a token does not end up in a committed file: export `HF_TOKEN` (or `LLMDBENCH_HF_TOKEN`). Gated models require it; public models need nothing. How the token is presented *inside* the cluster is a scenario matter -- `huggingface.secretName` and `huggingface.tokenKey`.

### What comes from the scenario file

Everything else. The single rule that shapes the file: **an engine flag is never a scenario key.** A role states its own launch command, verbatim, in the engine's own spelling; llm-d-benchmark reads that text for the handful of facts Kubernetes needs before the process starts (which engine, which port, how many accelerators, the model reference) and passes the rest through untouched. So there is no key for `--max-model-len`, `--tp-size` or `--kv-transfer-config`: they go in the command, exactly as the llm-d guides write them.

[`config/README.md`](../config/README.md) is the full key reference -- every key, its default, and what reads it. The areas standup draws on:

| Area                       | Keys                                                                                                                                                                              |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Engine launch              | `<role>.engine.command` (`decode`, `prefill`, `standalone`, `nok8s`), `standalone.engine.args`, `engine.preprocessScript`, `<role>.engine.port`, `engine.servicePort`, `engine.name` |
| Model                      | `model.name`, `model.shortName`, `model.size`, `modelservice.uriProtocol` (`pvc` \| `hf`)                                                                                          |
| Scale and hardware         | `<role>.replicas`, `<role>.resources`, `<role>.parallelism`, `accelerator.resource`, `accelerator.type`, `accelerator.memory`, `<role>.acceleratorType`, `affinity`                 |
| Storage                    | `storage.modelPvc.*`, `storage.workloadPvc.*`, `storage.downloadTimeout`, `storage.hostPath.*`, `standalone.modelMountPath`                                                        |
| Routing                    | `gateway.className`, `routing.proxy.enabled`, `routing.connector`, `router.epp.*`, `httpRoute.*`                                                                                   |
| Pod plumbing               | `<role>.extraEnvVars`, `<role>.extraContainerConfig`, `<role>.initContainers`, `<role>.additionalVolumes`/`additionalVolumeMounts`, `engine.volumes`/`engine.volumeMounts`, `engine.networkResource`, `engine.networkNr`, `engine.ephemeralStorage` |
| Images and charts          | `images.vllm`, `images.sglang`, `images.trtllm`, `images.llmdInferenceSim`, `chartVersions.*`, `helmRepositories.*`                                                                |
| Namespace and identity     | `namespace.name`, `serviceAccount.name`, `serviceAccountOverride`, `schedulerName`, `release`, `labels.*`                                                                          |
| Timing and validation      | `control.waitTimeout`, `control.ignoreFailedValidation`                                                                                                      |
| Monitoring                 | `monitoring.*`, `<role>.monitoring.podmonitor.*`                                                                                                                                   |
| Harness and workload       | `harness.name`, `harness.experimentProfile`, `harness.resources`, `workDir`                                                                                                        |

Anything a scenario does not state falls back to [`config/templates/values/defaults.yaml`](../config/templates/values/defaults.yaml). A good scenario reads as the difference between its deployment and that baseline; a value restated from defaults is one more thing that silently goes stale.

`control.ignoreFailedValidation` (default `true`) decides what happens when the capacity planner's sanity check -- the role's stated parallelism widths against the devices its pod requests, context length and KV cache against accelerator memory -- comes back unhappy: continue to deployment, or stop. It reads the context length and memory fraction back out of the engine command, so a scenario states them once, in the flag the engine itself reads.

On clusters where users cannot provision PersistentVolumeClaims, pass `standup --no-pvc` to avoid creating the model PVC. The workload PVC and data-access pod are a run-phase concern -- they first appear at `run`, and `run --no-pvc` skips them too. See [Standing up without PVCs](../llmdbenchmark/standup/README.md#standing-up-without-pvcs---no-pvc) for details.

Gateway class options (set via `gateway.className` in the scenario YAML):

| `className`                  | What it deploys                                                                                                  | Use when                                                          |
|------------------------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `none`                       | ModelService decode pods plus a plain ClusterIP Service; **no** Gateway, HTTPRoute, EPP, Envoy, or routing proxy | Measuring direct model-server performance and routing overhead    |
| `istio`                      | istio-base + istiod control plane, a Gateway + HTTPRoute, the `llm-d-router-gateway-dev` chart                       | Most flexible / production deployments                            |
| `agentgateway`               | agentgateway-crds + agentgateway controller, a Gateway + HTTPRoute, the `llm-d-router-gateway-dev` chart             | Want agentgateway's data plane instead of Envoy/Istio             |
| `gke`                        | Uses GKE-managed Gateway controller; same `llm-d-router-gateway-dev` chart                                           | Running on GKE                                                    |
| `data-science-gateway-class` | OpenDataHub / OpenShift AI managed Gateway                                                                           | Running on OpenShift AI                                           |
| `epponly` (default)          | **No** Kubernetes Gateway, **no** HTTPRoute, the `llm-d-router-standalone-dev` chart (EPP with an Envoy sidecar serving HTTP) | Default; llm-d's standalone router topology, no gateway needed     |

`none` is a baseline lane, not a routing topology. It requires at least one
decode replica and does not support P/D disaggregation.

### Overriding `gateway.className` from the CLI

Every subcommand that renders templates (`plan`, `standup`, `experiment`,
and the `run`/`smoketest`/`teardown` paths that re-render for setup
overrides) accepts a `--gateway-class` flag that overrides the
scenario's `gateway.className` for that invocation. The same value can
be supplied via the `LLMDBENCH_GATEWAY_CLASS` environment variable.

```bash
# Scenario default is epponly -- flip to istio without editing YAML
llmdbenchmark --spec guides/optimized-baseline standup -p my-ns --gateway-class istio

# env-var form (matches the LLMDBENCH_* convention)
LLMDBENCH_GATEWAY_CLASS=agentgateway \
  llmdbenchmark --spec guides/optimized-baseline standup -p my-ns
```

Precedence (highest wins): `--gateway-class` CLI flag → scenario
`gateway.className` → `defaults.yaml` (`epponly`).

#### Method-aware validation

`gateway.className` only affects rendering when the active deploy
method is `modelservice`. For `kustomize`, `standalone`, and `fma` the
gateway block is ignored by every rendered template, so the CLI accepts
**any** value (including sentinels like `none` or `n/a` that CI scripts
often pass uniformly across deploy methods). The banner shows it as
`Gateway: <value> (ignored -- modelservice is not the active deploy method)`.

When `modelservice` is the active method, the override is checked
against the whitelist of supported values above and a typo fails fast
at plan time:

```text
ValueError: --gateway-class='isto' is not a supported value for the
modelservice deploy method. Choose one of: epponly, istio, agentgateway,
gke, data-science-gateway-class.
```

This lets CI workflows pass `--gateway-class=$SOME_VAR` for every
deploy method without per-method special-casing, while still catching
typos when the value actually matters.

### Switching from Istio to agentgateway

By default, `llm-d-benchmark` deploys [Istio](https://istio.io/) as the gateway provider for the `modelservice` deployment method.  To use [agentgateway](https://agentgateway.dev/) instead, add a `gateway` block to your scenario YAML:

```yaml
scenario:
  - name: "my-stack"
    gateway:
      className: agentgateway       # default is "istio"

    modelservice:
      enabled: true
    # ... rest of scenario config
```

That single change is all that's needed.  The benchmark tool handles everything else automatically:

1. **Installs agentgateway** -- the controller and CRDs are installed via helmfile during step 02 (admin prerequisites), the same way Istio is installed
2. **Configures the Gateway resource** -- the llm-d-infra Helm chart creates a `Gateway` with `gatewayClassName: agentgateway`
3. **OpenShift SCC** -- on OpenShift clusters, a minimal custom SCC (`llmdbench-agentgateway`) is automatically created and granted to the gateway service account, allowing the proxy to run as UID 10101 with `NET_BIND_SERVICE`

#### Differences from Istio

| Aspect                    | Istio                                                  | agentgateway                                                          |
|---------------------------|--------------------------------------------------------|-----------------------------------------------------------------------|
| Gateway pod creation      | Created by the llm-d-infra Helm chart directly         | Created dynamically by the agentgateway controller                    |
| `gatewayParameters`       | Uses `ConfigMap`-based `parametersRef`                 | Not used -- agentgateway manages its own `AgentgatewayParameters` CRD |
| OpenShift compatibility   | Built-in via `floatingUserId` (uses namespace UID range) | Requires custom SCC (auto-created by the tool)                      |
| Service name              | `infra-{release}-inference-gateway-istio`              | `infra-{release}-inference-gateway`                                   |

#### Example scenarios using agentgateway

- [`config/scenarios/examples/cpu.yaml`](../config/scenarios/examples/cpu.yaml) -- CPU-only deployment
- [`config/scenarios/guides/optimized-baseline.yaml`](../config/scenarios/guides/optimized-baseline.yaml) -- inference scheduling guide

### EPP-only (llm-d standalone router) mode (`gateway.className: epponly`)

`epponly` mirrors the **Standalone Mode** documented in llm-d
([guides/recipes/router/README.md](https://github.com/llm-d/llm-d/blob/main/guides/recipes/router/README.md))
and used by every well-lit-path guide (e.g.
[optimized-baseline](https://github.com/llm-d/llm-d/blob/main/guides/optimized-baseline/README.md)).
The EPP is deployed via the llm-d-owned
`oci://ghcr.io/llm-d/charts/llm-d-router-standalone-dev`
chart (migrated from the upstream GAIE-published `standalone` chart),
which adds an Envoy sidecar to the EPP pod so HTTP traffic can hit the
EPP service directly -- no Kubernetes Gateway, no HTTPRoute, no
`llm-d-infra` Helm release.

```yaml
scenario:
  - name: "my-stack"
    gateway:
      className: epponly      # default is "istio"

    modelservice:
      enabled: true
    # ... rest of scenario config
```

When `epponly` is selected, standup automatically:

1. **Skips the gateway provider install** -- step 02 does not install istio
   or agentgateway CRDs / controllers.
2. **Skips the `llm-d-infra` Helm release** -- no Gateway resource is created.
3. **Skips HTTPRoute rendering** -- nothing references a Gateway.
4. **Swaps the router chart** to the `llm-d-router-standalone-dev` chart,
   which bundles the EPP + Envoy sidecar in a single pod.
5. **Adds a `port 80 -> targetPort 8081` extraServicePort** to the EPP
   service so HTTP requests reach the Envoy sidecar.
6. **Points endpoint discovery at `{model_id_label}-router-epp:80`** -- the
   smoketest and run phase resolve to the EPP service directly instead
   of a Gateway IP.

#### Restrictions

- **Single-stack only.** `epponly` cannot multiplex multiple models since
  it has no shared Gateway / HTTPRoute. Multi-stack scenarios fail at
  render time with a clear error.
- **`httpRoute.mode: shared` is rejected** -- shared HTTPRoute requires a
  Gateway that `epponly` does not deploy.
- **Only the `modelservice` deploy method.** Combining `epponly` with
  `standalone.enabled: true`, `fma.enabled: true`, or
  `kustomize.enabled: true` is rejected at render time.

#### Differences from a gateway-based deployment

| Aspect                   | Gateway-based (istio / agentgateway / gke)             | `epponly`                                                    |
|--------------------------|--------------------------------------------------------|--------------------------------------------------------------|
| Router Helm chart        | `llm-d-router-gateway-dev`                             | `llm-d-router-standalone-dev`                                |
| `llm-d-infra` release    | Installed (creates `Gateway`)                          | **Skipped** (no Gateway needed)                              |
| HTTPRoute                | Rendered                                               | **Not rendered**                                             |
| Provider control plane   | istio / agentgateway controller installed via helmfile | **Not installed**                                            |
| Endpoint                 | `Gateway` resource IP                                  | `{model_id_label}-router-epp` Service ClusterIP, port 80     |
| Number of EPP replicas   | Configurable                                           | **1** (matches default `router.epp.replicas: 1`)             |
| Multi-stack support      | Yes                                                    | **No** (single-stack only)                                   |

#### Example scenario using epponly

- [`config/scenarios/guides/optimized-baseline.yaml`](../config/scenarios/guides/optimized-baseline.yaml)
  -- ships with `gateway.className: epponly` as the default. Flip it to
  `istio`/`agentgateway`/`gke`/`data-science-gateway-class` to switch
  topology without touching anything else in the scenario.

### Keys for the llm-d charts themselves

Standup deploys the llm-d charts -- modelservice, the router/endpoint picker, and (for gateway-backed topologies) the infra chart -- and every knob on them is a scenario key. The ones scenarios reach for most:

| Key                                       | Meaning                                                                                                          |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `modelservice.enabled`                    | Deploy via the modelservice chart (`decode`/`prefill` roles). The alternative lanes are `standalone`, `kustomize`, `nok8s` and `fma`. |
| `modelservice.uriProtocol`                | `pvc` (default: a download Job stages the weights and the engine reads them from the PVC) or `hf` (the chart hands the engine an `hf://` artifact and it pulls at start). Read at this level -- under `storage:` it parses and nothing reads it. |
| `decode.engine.port`, `prefill.engine.port` | Override the port a role binds. Normally omit: it is read from the command, and the default follows the topology -- decode binds 8200 behind the routing sidecar, 8000 without it; prefill and standalone bind 8000. |
| `gateway.className`                       | Router topology -- see the table above.                                                                          |
| `gateway.name`, `gateway.namespace`, `gateway.logLevel`, `gateway.service.type`, `gateway.resources` | The Gateway object and its data plane, for the gateway-backed classes. |
| `router.epp.replicas`, `router.epp.resources`, `router.epp.env`, `router.epp.verbosity` | The endpoint picker pod.                                        |
| `router.epp.pluginsConfigFile`, `router.epp.pluginsCustomConfig` | Which scheduling plugins the EPP runs, and their inline config. Write these under `router.epp` -- one level up, under `router`, they render into chart values that nothing reads and the EPP falls back to its default config. |
| `router.tokenizer.enabled`                | Run the UDS tokenizer sidecar alongside the EPP.                                                                 |
| `router.inferencePool.failureMode`        | `FailOpen` (default) or `FailClose` when the EPP is unreachable.                                                  |
| `router.monitoring.prometheus.enabled`, `router.monitoring.interval` | ServiceMonitor for EPP metrics.                                        |
| `routing.proxy.enabled`                   | Whether decode pods get the routing sidecar. This decides the decode port.                                        |
| `routing.connector`, `routing.secure`, `routing.debugLevel` | KV transfer connector and sidecar behaviour for P/D disaggregation.               |
| `httpRoute.requestTimeout`, `httpRoute.backendRequestTimeout` | HTTPRoute timeouts, for the gateway-backed classes.                            |
| `chartVersions.llmDModelservice`, `chartVersions.llmDRouter`, `chartVersions.llmDInfra`, `chartVersions.inferencePool` | Chart versions. `auto` resolves to the newest published release at standup time; pin a version to make a run reproducible. |
| `helmRepositories.*`                      | Where those charts are fetched from.                                                                              |
| `images.routerEndpointPicker`, `images.routingSidecar`, `images.udsTokenizer` | Override an llm-d component image.                             |

Full list, with defaults, in [`config/README.md`](../config/README.md) and [`config/templates/values/defaults.yaml`](../config/templates/values/defaults.yaml).
