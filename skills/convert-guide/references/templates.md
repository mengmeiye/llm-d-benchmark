# Scenario File Skeleton

One file per guide, at `config/scenarios/guides/<guide-name>.yaml`. Everything
below is optional except `name`, one role, and that role's command -- leave out
whatever equals the default in `config/templates/values/defaults.yaml`.

```yaml
# ============================================================================
# <GUIDE NAME>
# Converted from https://github.com/llm-d/llm-d/tree/main/guides/<guide-name>
#
# Not carried over:
# - <thing>  (<why, or where to add it back>)
# ============================================================================

scenario:
  - name: "<guide-name>"

    # -----------------------------------------------------------------------
    # COMMON -- shared by every role. Hoisted to the config root, so
    # `common.model` and a top-level `model:` are the same thing.
    # -----------------------------------------------------------------------
    common:
      model:
        # Kubernetes name prefix for this stack's Deployments, Services, PVCs.
        # The model id itself comes from the launch command, not from here.
        shortName: <short-slug>
        size: 1Ti

      storage:
        modelPvc:
          size: 1Ti
          # storageClassName: standard-rwx

      # Only when the guide pins a non-default image. One entry per engine:
      # vllm / sglang / trtllm / llmdInferenceSim. A single role can override
      # with `<role>.engine.image`.
      # images:
      #   vllm:
      #     repository: docker.io/vllm/vllm-openai
      #     tag: v0.12.0

      # Pod shape, not engine flags: the shell the command is exec'd in, the
      # preprocess step, volumes, the Service port, RDMA devices.
      engine:
        preprocessScript: ". /shared-config/llmdbench_env.sh"
        volumes:
          - name: dshm
            type: emptyDir
            emptyDir:
              medium: Memory
              sizeLimit: 16Gi
        volumeMounts:
          - name: dshm
            mountPath: /dev/shm

    # -----------------------------------------------------------------------
    # MODELSERVICE -- the llm-d deployment. `gateway`, `router`, `routing`,
    # `httpRoute`, `multinode`, `prefill` and `decode` may be nested here (for
    # readability) or written at the top level; both resolve the same.
    # -----------------------------------------------------------------------
    modelservice:
      enabled: true

      # `pvc+hf` (default) stages a Hugging Face hub cache on a PVC via a
      # download Job and points the pod's HF_HUB_CACHE at it, so the command's
      # plain model id resolves locally; `hf` has the engine pull the same id at
      # pod start, no PVC; `pvc` stages a raw weights directory the engine is
      # given as a path. Note the level: `modelservice.uriProtocol`.
      # uriProtocol: hf

      gateway:
        className: epponly        # or istio / agentgateway / gke

      routing:
        proxy:
          enabled: true           # the decode sidecar; decides decode's port

      router:
        epp:
          replicas: 1
          flags:
            v: 2
          # pluginsConfigFile must name a key in pluginsCustomConfig.
          pluginsConfigFile: "<guide-name>-plugins.yaml"
          pluginsCustomConfig:
            <guide-name>-plugins.yaml: |
              apiVersion: llm-d.ai/v1alpha1
              kind: EndpointPickerConfig
              plugins: ...

      prefill:
        enabled: false            # true for a P/D guide
        replicas: 0

      decode:
        replicas: 2
        engine:
          # The guide's launch command, verbatim. 8200 behind the routing
          # sidecar, 8000 without it.
          command: |
            vllm serve <model-id> \
            --host 0.0.0.0 \
            --port 8200 \
            --tensor-parallel-size 2 \
            --max-model-len 16000 \
            --gpu-memory-utilization 0.9
        resources:
          limits:
            memory: 128Gi
            cpu: "16"
          requests:
            memory: 64Gi
            cpu: "8"
        extraEnvVars:
          - name: <NAME>
            value: "<value>"
        # extraContainerConfig:
        #   securityContext: ...

    # -----------------------------------------------------------------------
    # WORKLOAD -- not in the guide; from the user's request or the defaults.
    # -----------------------------------------------------------------------
    harness:
      name: inference-perf
      experimentProfile: sanity_random.yaml

    workDir: "~/data/<guide-name>"
```

## Smallest useful file

A guide with one decode role, default everything else, and no PVC:

```yaml
scenario:
  - name: "minimal-guide"
    common:
      model:
        shortName: qwen-qwen3-0-6b
        size: 20Gi
    modelservice:
      enabled: true
      uriProtocol: hf
      decode:
        replicas: 1
        engine:
          command: |
            vllm serve Qwen/Qwen3-0.6B \
            --host 0.0.0.0 \
            --port 8200 \
            --max-model-len 32768
    harness:
      name: inference-perf
      experimentProfile: sanity_random.yaml
    workDir: "~/data/minimal-guide"
```

## Kustomize passthrough

For a guide with a `kustomization.yaml`, the whole conversion can be:

```yaml
scenario:
  - name: "<guide-name>"
    kustomize:
      enabled: true
      guideName: "<guide-dir-name>"
      acceleratorBackend: "gpu/vllm"
      guideVariableOverrides: {}
    harness:
      name: inference-perf
      experimentProfile: sanity_random.yaml
    workDir: "~/data/<guide-name>"
```

## Render check

```bash
llmdbenchmark --spec guides/<guide-name> standup --dry-run
```

Read the output, do not just check the exit code. The files that matter:

| Rendered file | What to confirm |
|---|---|
| `13_ms-values.yaml` | the command is the guide's text; `modelArtifacts.uri` is `hf://` or `pvc://` as intended; the container port and probes match the command's `--port` |
| `12_router-values.yaml` | your plugin config is under `router.epp.pluginsConfigFile` / `pluginsCustomConfig`, not one level up |
| `config.yaml` | `model.maxModelLen` and `gpuMemoryUtilization` match the command's flags; each role's accelerator request matches the width its command asks the engine for |
| `14_standalone-deployment_yaml.yaml` | same checks, for a standalone scenario |

No `${...}` should survive in a rendered command unless you put it there on
purpose (`${model.blockSize}`).

## Experiment files

A scenario stands one stack up. Sweeping across treatments is a separate
top-level file, `experiments/<name>.yaml`, with `experiment:` (name,
description, harness, profile) and `design:` (`type: full_factorial`, `setup:`)
blocks, run as:

```bash
llmdbenchmark --spec guides/<guide-name> experiment --experiments experiments/<name>.yaml
```

Conversion does not produce one. Copy the closest existing file --
`experiments/optimized-baseline.yaml` is the fullest -- if the user asks for a
sweep.
