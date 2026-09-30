# Conversion Patterns

The default conversion is one role, one command (see SKILL.md). This file covers
the guides that are not that. Everything here is still the same rule: the
command is copied, and only the Kubernetes facts around it are written as keys.

## Which engine -- usually nothing to write

The launcher in the command text selects the engine, and the engine selects the
image, the health path and the metrics path:

| Command starts with | Engine | Image key |
|---|---|---|
| `vllm serve` | vllm | `images.vllm` |
| `python3 -m sglang.launch_server` | sglang | `images.sglang` |
| `trtllm-serve serve` | trtllm | `images.trtllm` |
| `llm-d-inference-sim` | sim | `images.llmdInferenceSim` |

So `engine.name` is normally absent from a scenario. Write it in two cases:

- **The image launches the server itself**, so there is no command to detect.
  Then `engine.name` is the only way to pick the image and paths.
- **An engine with no spec.** `engine.name: generic` runs the command verbatim
  and derives nothing from it, so the scenario must then state
  `<role>.engine.port` by hand. (`<role>.parallelism` is stated in every case --
  it is never read from a command.)

If you do write `engine.name` and it disagrees with the command's launcher, the
resolver trusts the command and warns. That warning means one of the two is
wrong -- do not silence it by deleting the command's launcher.

Each engine's flag spellings are in `llmdbenchmark/engine/spec.py`: SGLang's
`--tp-size`, `--context-length`, `--page-size`, `--mem-fraction-static` and
TRT-LLM's `--tp_size`, `--max_seq_len`, `--tokens_per_block` are read as the
same facts as vLLM's. You do not need to know this to convert a guide -- copy
the guide's spelling -- but it is why you never translate flags between engines.

## An image that launches itself

`llm-d-inference-sim` v0.9+ is distroless: no shell to exec a command in. State
an empty command and pass `args:` instead. An empty string, not a null -- a YAML
key with no value leaves the default in place.

```yaml
      engine:
        command: ""
        args:
          - "--model"
          - "/model-cache/${model.path}"
          - "--port"
          - "8000"
          - "--served-model-name"
          - "facebook/opt-125m"
```

See `config/scenarios/examples/sim.yaml`.

## Something must run before the engine

Do not prepend it to the command. Two homes, by lifetime:

- `engine.preprocessScript` -- runs in the *same* container and shell, just
  ahead of the command. This is where an env-file `source`, a `ulimit`, a
  `LD_LIBRARY_PATH` export or a cache-dir `mkdir` goes. The default already
  sources the shared config: `. /shared-config/llmdbench_env.sh`.
- `<role>.initContainers` -- a separate container that must finish first
  (writing the shared config, warming a cache, staging weights). Copy the
  guide's init containers into this list as they are written.

A command that legitimately needs several statements -- writing a config file
the engine then reads -- can hold them, separated by `;`, because the whole
thing runs in one shell. The TensorRT-LLM alternative in
`config/scenarios/examples/engines.yaml` does exactly that to produce
`--extra_llm_api_options`.

## P/D disaggregation -- two roles, two commands

```yaml
    modelservice:
      routing:
        connector: nixlv2
      prefill:
        enabled: true          # off by default
        replicas: 1
        engine:
          command: |
            vllm serve ... --port 8000 ...
      decode:
        replicas: 2
        engine:
          command: |
            vllm serve ... --port 8200 ...
```

Points that are easy to get wrong:

- **Ports differ by role, not by preference.** Decode sits behind the routing
  sidecar, which owns the Service's 8000, so decode binds 8200. Prefill never
  gets a sidecar, so it binds 8000 -- which is also where decode's transfer
  reaches it. With `routing.proxy.enabled: false` decode binds 8000 too.
- **The KV connector flag goes in both commands**, copied from the guide, e.g.
  `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`.
  Quote it exactly as the guide does; it is opaque to llm-d-benchmark.
- **The two roles may have different parallelism**, so each one states its own
  `parallelism` block to match the width in its own command.

`config/scenarios/guides/pd-disaggregation.yaml` is the worked example.

## Multi-node (LeaderWorkerSet)

A model that does not fit one node is deployed as an LWS group -- one leader
plus workers, scheduled and scaled together:

```yaml
    modelservice:
      multinode:
        enabled: true
      decode:
        parallelism:
          tensor: 1
          data: 1
          dataLocal: 8      # accelerators per node
          workers: 2        # nodes per replica
```

`workers` is the LWS group size; `dataLocal` (or `tensor`) is the per-node
width. Their product is the group's accelerator count. `workers` is the one width
with no counterpart in the command at all: how many *nodes* to spread a replica
over is a Kubernetes decision, not an engine flag.

If the guide ships a hand-written `LeaderWorkerSet` manifest, do not try to
express it as keys. Either take the kustomize path (SKILL.md Step 1) or record
in the scenario header that the raw LWS resource was not carried over.

`config/scenarios/guides/wide-ep.yaml` is the worked example.

## Endpoint-picker plugin configuration

The guide's `inferenceExtension.pluginsCustomConfig` is a whole YAML document
(an `EndpointPickerConfig`: plugins, then a schedulingProfiles section). Copy it
in full, keyed by the same filename the guide uses:

```yaml
    modelservice:
      router:
        epp:
          pluginsConfigFile: "my-guide-config.yaml"
          pluginsCustomConfig:
            my-guide-config.yaml: |
              apiVersion: llm-d.ai/v1alpha1
              kind: EndpointPickerConfig
              plugins:
                - type: prefix-cache-scorer
                  parameters:
                    blockSize: ${model.blockSize}
              ...
```

Do not summarise it, do not reorder the plugins, and do not drop a plugin you
do not recognise -- the file is the scheduler's whole behaviour and a missing
entry changes results silently. `pluginsConfigFile` must name a key that exists
in `pluginsCustomConfig`.

Where the config states a KV page size, it and the engine's command must agree
on the same number. That is what `${model.blockSize}` is for: put it in both
places rather than the literal. It is the one substitution worth the noise.

Watch the nesting: `pluginsConfigFile`, `pluginsCustomConfig` and
`resources` belong under `router.epp`. Placed one level up, under `router`,
they render into the chart values where nothing reads them -- the EPP silently
runs the default plugin config. The rendered `12_router-values.yaml` is where
you catch this: look for your filename under `router.epp.pluginsConfigFile`,
not under `router.`.

Other EPP keys come from `gaie-*/values.yaml`: `replicas`, `flags`, `env`,
`resources`. `router.tracing`, `router.modelServers`, `router.proxy` and
`router.inferencePool` do sit at the `router` level.

## An accelerator that needs a different command

XPU takes no `--block-size` and wants `--enforce-eager`; Spyre and CPU differ
again. Do not parameterise one command to cover them. Write a separate scenario
file with the flags stated plainly -- `config/scenarios/examples/intel-xpu.yaml`,
`examples/spyre.yaml`, `examples/cpu.yaml`.

What an accelerator *profile* contributes is values, never command text: the
image, resource sizing, storage and router sizing, in
`config/templates/values/overlays/<name>.yaml`, auto-detected from the cluster.
So a per-accelerator scenario only needs its command and whatever the overlay
does not already set. It must not name a model that the overlay's hardware
cannot hold, and it must never contain command fragments for other backends.

## Kustomize guides

A guide with a `kustomization.yaml` can be applied as-is:

```yaml
    kustomize:
      enabled: true
      guideName: "<guide-dir-name>"
      acceleratorBackend: "gpu/vllm"     # or gpu/sglang
      guideVariableOverrides: {}          # fills the README's ${VAR}s
      patches: []
```

This is usually the better answer: the guide's own manifests deploy, so nothing
can be lost in translation, and the only conversion left is the `harness:`
block. Offer it before converting by hand.

## Extra env, volumes, and container fields

| In the guide | Scenario key |
|---|---|
| container `env:` | `<role>.extraEnvVars` -- every entry, name and value, including ones whose purpose is unclear |
| pod-level volumes (`dshm`, `shared-config`) | `engine.volumes` / `engine.volumeMounts` |
| role-specific volumes | `<role>.additionalVolumes` / `<role>.additionalVolumeMounts` |
| `/dev/shm` size | the `dshm` entry in `engine.volumes`: `emptyDir.sizeLimit`. (Scenarios also carry an `engine.shmMemory` key; nothing reads it, so the `sizeLimit` is the one that matters.) |
| RDMA/IB devices | `engine.networkResource`, `engine.networkNr` |
| `securityContext`, extra `ports`, `imagePullPolicy`, anything else | `<role>.extraContainerConfig` |
| a whole extra Kubernetes object | `extraObjects` |

`extraEnvVars` is where conversions lose the most: a guide's env block is easy
to skim past and its absence usually shows up as a runtime failure, not a
render error. Diff the guide's env list against the scenario's before reporting
done.

## Standalone (no llm-d)

A guide that deploys a bare engine Deployment with no gateway or endpoint picker
converts to `standalone:` with `modelservice.enabled: false`. Same command
rule; standalone always binds 8000. `config/scenarios/examples/sim.yaml` shows
the block's shape and `config/scenarios/examples/launcher.yaml` is a live one.
`config/scenarios/guides/nok8s.yaml` (`nok8s.enabled: true`) is the same idea
with no cluster at all.
