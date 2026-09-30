---
name: convert-guide
description: Convert an llm-d deployment guide into an llm-d-benchmark scenario file. Use when the user points at an llm-d guide (Helm values or kustomize) and wants a runnable scenario. Triggers on /convert-guide with a URL or local path.
---

# Convert an llm-d Guide to a Scenario File

## What this conversion is

An llm-d guide and an llm-d-benchmark scenario describe the same deployment. The
guide says it in Helm values; a scenario says it in one YAML file. The engine's
launch command is the same text in both places, so **the conversion copies it,
it does not translate it.**

That is the whole design. There is no table mapping `--max-model-len` to a
parameter, because llm-d-benchmark does not have a parameter for it: the command
is passed to the engine as written, and the few facts Kubernetes needs before
the engine starts are *read back out* of it. So your job is:

1. find the engine container in the guide,
2. copy its command verbatim into `<role>.engine.command`,
3. write down the handful of things that are **not** in the command,
4. name the stack.

If you find yourself converting a flag into a YAML key, stop -- that flag
belongs in the command.

```
/convert-guide <url-or-path>
/convert-guide <url-or-path> with <harness> <profile>
```

Defaults: harness `inference-perf`, profile `sanity_random.yaml`. See
[references/harnesses.md](references/harnesses.md).

## Step 1 -- Read the guide

Detect the shape first:

- **`kustomization.yaml` present** -> kustomize guide. llm-d-benchmark can apply
  these manifests *verbatim* instead of converting them: set
  `kustomize.enabled: true` and `kustomize.guideName`, and the guide's own
  topology is used as-is. Offer this to the user; it is usually the better
  answer for a kustomize guide, and the only conversion work left is the
  `harness:` block. If they want a real conversion, read the base plus the
  `op: replace` / `op: add` patches and continue below.
- **`ms-*/values.yaml` present** -> Helm values guide. Read `ms-*/values.yaml`
  (ModelService), `gaie-*/values.yaml` (the endpoint picker) and
  `helmfile.yaml.gotmpl` if present (chart versions).

Record file paths and line numbers as you go -- the scenario's comments should
say where each non-obvious value came from.

Do not read `config/scenarios/guides/*.yaml` looking for *mappings*; they are
outputs. Do read one as a **shape** reference --
`config/scenarios/guides/precise-prefix-cache-routing.yaml` is the fullest
example (router plugins, init container, extra env, multi-role).

## Step 2 -- Copy each role's command verbatim

In the guide, each role (decode, prefill) has a container with a `command` /
`args` that starts the engine. Copy that text into the scenario:

```yaml
      decode:
        replicas: 2
        engine:
          command: |
            vllm serve Qwen/Qwen3-32B \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 2 \
            --max-model-len 16000 \
            --gpu-memory-utilization 0.95
```

Copy it whole, including flags you do not recognise, JSON blobs
(`--kv-transfer-config`, `--kv-events-config`) and `$(ENV_VAR)` references --
Kubernetes expands those in the pod. Keep the guide's own line order and
spelling; a reviewer should be able to diff the two.

The same rule holds for every engine. `python3 -m sglang.launch_server
--model-path ...` and `trtllm-serve serve ...` are copied exactly as the guide
writes them; the launcher in the text is what selects the engine, so
`engine.name` is rarely needed (see `config/scenarios/examples/engines.yaml`,
which carries all three).

**Three adjustments, and only these three:**

1. **The port.** A decode pod that sits behind the routing sidecar
   (`routing.proxy.enabled: true`, the default) must bind **8200** -- the
   sidecar holds the Service's 8000. With the sidecar off, decode binds 8000.
   Prefill and standalone always bind 8000. If the guide's number disagrees with
   the topology you are writing, fix the number and say why in a comment.
2. **The model reference.** Write the model id literally
   (`vllm serve Qwen/Qwen3-32B`) -- that is what makes the line paste-and-go, and
   `model.name` is read off it. Drop `--served-model-name` when it merely repeats
   the serve target: every engine advertises whatever it was asked to serve.
   `${model.name}` replaces the literal in exactly two cases -- a scenario meant
   to be swept with `-m/--models`, and one command shared by stacks that serve
   different models. `uriProtocol: pvc` is the one protocol where the serve
   target is a mounted directory rather than an id; there, keep a *literal*
   `--served-model-name`, which is where the id is then read from.
3. **Values two processes must agree on.** Where the guide's page size is also
   read by the endpoint picker's token processor, keep the engine's flag literal
   and point the picker at `${model.blockSize}` -- that value is read back off
   the command, so there is one number and it is the engine's own. This is the
   rare case; everything else stays literal.

Leave in place anything else the guide's command does. In particular do **not**
add a preprocess step, a shell prelude, an env-file `source`, or a library path
to the command. If the deployment needs one, it goes in
`engine.preprocessScript`, which runs ahead of the command in the same shell.

## Step 3 -- Write down what the command does not say

Everything here is a Kubernetes or workload fact, not an engine flag:

| Scenario key | From the guide |
|---|---|
| `model.shortName` | Not in the guide. You choose it: a readable slug of the model id (`qwen-qwen3-32b`). It prefixes Deployments, Services and PVCs, so keep it short and stable. |
| `model.size`, `storage.modelPvc.size` | Weight size; large enough for the model. |
| `modelservice.uriProtocol` | `pvc+hf` (default: a download job stages a Hugging Face hub cache on a PVC and the pod's `HF_HUB_CACHE` points at it), `hf` (engine pulls at start, no PVC), or `pvc` (a raw weights directory the engine is given as a path). Note the level -- under `storage:` it parses fine and nothing reads it. |
| `<role>.replicas` | `replicas` on the guide's decode/prefill stage. |
| `<role>.resources` | The container's `memory` and `cpu`. An accelerator count belongs here only when `<role>.parallelism` cannot express it -- `limits.<accelerator resource>` wins over every shorthand. |
| `<role>.parallelism` | The widths as llm-d chart values, which is also what sizes the pod's accelerator request. Write them whenever the role is not single-device: the kubelet grants accelerators before the engine exists, so they are **not** read out of the command, and they have to agree with the width the command gives the engine (`--tensor-parallel-size 4` -> `tensor: 4`). |
| `<role>.extraEnvVars` | The container's `env:`, every entry. |
| `<role>.extraContainerConfig` | `securityContext`, extra `ports`, `imagePullPolicy` -- container config with no first-class key. |
| `engine.volumes` / `engine.volumeMounts` | Pod-level volumes (`dshm`, `shared-config`). Per-role ones go in `<role>.additionalVolumes` / `additionalVolumeMounts`. |
| `engine.shmMemory`, `engine.networkResource`, `engine.networkNr` | `/dev/shm` sizing and RDMA/IB devices. |
| `modelservice.gateway.className` | The guide's router topology: `epponly` (no Gateway), `istio`, `agentgateway`, `gke`. |
| `modelservice.routing.proxy.enabled` | Whether the guide's decode pod has a routing sidecar. This decides the decode port -- see Step 2. |
| `router.epp.*` | From `gaie-*/values.yaml`: `replicas`, `flags`, `env`, `resources`. |
| `router.epp.pluginsCustomConfig` | `inferenceExtension.pluginsCustomConfig` -- copy the **entire** embedded YAML document, not a summary. |
| `images.<engine>` | Only if the guide pins a non-default image. Keys are named per engine: `vllm`, `sglang`, `trtllm`, `llmdInferenceSim`. |
| `chartVersions.*` | Release versions from the helmfile, if pinned. |
| `harness.name`, `harness.experimentProfile` | Not in the guide -- from the user's request, or the defaults. |
| `workDir` | `~/data/<guide-name>` |

Leave out anything equal to the default in
`config/templates/values/defaults.yaml`. A scenario should read as the
difference between this deployment and the baseline, and every value restated
from defaults is one more thing that silently goes stale.

Two things that look like they belong here but must not be restated: the capacity
pair (`maxModelLen`, `gpuMemoryUtilization`) and the model id. The resolver reads
all three off the command -- the pair for the pre-deploy capacity check, the id
for the PVC, the pod labels and the HTTPRoute. Writing them in the `model:` block
too means a future edit to the command leaves the check reasoning about a pod that
was never launched.

The engine's other numbers -- batch widths, page size, parallelism widths -- have
no `model:` key at all; they stay in the command, where the engine reads them. The
one Kubernetes fact that must be stated alongside a width is the device count: see
[references/patterns.md](references/patterns.md).

See [references/patterns.md](references/patterns.md) for the fiddly cases:
multi-node / LeaderWorkerSet, P/D disaggregation, endpoint-picker plugin
configs, accelerators that need a different command, and engines whose image
launches itself.

## Step 4 -- Write the file

Path: `config/scenarios/guides/<guide-name>.yaml`. Skeleton in
[references/templates.md](references/templates.md).

Head the file with the guide URL and a short list of anything you deliberately
did not carry over, so the next reader can tell an omission from an oversight:

```yaml
# ============================================================================
# <GUIDE NAME>
# Converted from https://github.com/llm-d/llm-d/tree/main/guides/<name>
#
# Not carried over:
# - pod monitoring (add via decode.extraContainerConfig)
# - torch-compile cache volume (not needed for this model/hardware)
# ============================================================================
```

You MUST actually write the file before reporting success.

## Step 5 -- Render it and read the output

A scenario that has never been rendered is a guess. Run:

```bash
llmdbenchmark --spec guides/<guide-name> standup --dry-run
```

Then check:

- **It renders with no errors.** Advisory warnings are acceptable; errors are not.
- **The command survived.** Find your role's command in the rendered
  `13_ms-values.yaml` (or `14_standalone-deployment_yaml.yaml`) and compare it
  to the guide's. It should differ only by the port and model-reference choices
  from Step 2.
- **The port matches the topology.** Container port, probes and the Service's
  target port all agree with the number in the command.
- **The router got your plugin config.** If you wrote one, the rendered
  `12_router-values.yaml` should show it at `router.epp.pluginsConfigFile` and
  `router.epp.pluginsCustomConfig`. One level up, under `router`, it renders
  into the chart values where nothing reads it and the EPP runs the default
  config instead.
- **The accelerator request is what you expect.** It follows `<role>.parallelism`
  (or `resources.limits.<accelerator resource>`); a wrong number means a width is
  missing there, and the engine will starve at startup asking for devices the pod
  was never granted.
- **The capacity reads landed.** `model.maxModelLen` and `gpuMemoryUtilization` in
  the rendered `config.yaml` should match the command's flags. `blockSize` is not
  read -- it only appears if the scenario stated it.
- **No `${...}` is left in the command** unless you put it there deliberately
  (Step 2, case 3). `${dotted.path}` resolves at render time -- one surviving in
  the output is a typo that will reach the engine as literal text.

## Report

Say where the file is, which guide files you read, what the engine command is,
and -- most importantly -- **what you did not carry over and why**. Silently
dropping guide configuration is the failure mode of this conversion; an explicit
list is how the user catches it.

## Reference files

- [references/harnesses.md](references/harnesses.md) -- harnesses and workload profiles
- [references/patterns.md](references/patterns.md) -- multi-node, P/D, plugins, non-CUDA accelerators
- [references/templates.md](references/templates.md) -- scenario skeleton
- `config/templates/values/defaults.yaml` -- every key and its default
- `config/README.md` -- the engine-command contract, in full
