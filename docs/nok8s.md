# No-Kubernetes (`nok8s`) deployment

## Concept

The `nok8s` deployment method runs the llm-d routing stack — **vLLM + EPP
(router) + Envoy** — as plain `docker`/`podman` containers on a single host,
with **no Kubernetes cluster**. The benchmark harness (`llmdbenchmark run`)
also runs as a **local container** against the Envoy front door, so the entire
standup → run → teardown lifecycle is cluster-free.

```
client ──▶ Envoy :8081 ──ext_proc──▶ EPP :9002 ──▶ picks a worker
              │                        (reads endpoints.yaml, file-discovery)
              └──────────────────────▶ vLLM :8000  (OpenAI-compatible API)
```

Instead of watching a Kubernetes `InferencePool`, the EPP reads its worker
inventory from a YAML file via the file-discovery plugin. This is the harness
equivalent of the upstream
[llm-d no-Kubernetes guide](https://github.com/llm-d/llm-d/tree/main/guides/no-kubernetes-deployment).

Use it for HPC/Slurm nodes, bare-metal boxes, or a single GPU workstation
where standing up Kubernetes is not worth it.

## Prerequisites

`llmdbenchmark standup` validates these automatically in step 00; a missing or
broken container runtime is fatal, the rest are warnings.

| Requirement | Notes |
|-------------|-------|
| Linux host + NVIDIA GPU(s) | Images/flags are NVIDIA + vLLM specific |
| NVIDIA driver | `nvidia-smi` must work |
| Container runtime | **docker** or **podman** (`nok8s.runtime`) |
| NVIDIA Container Toolkit | docker: `nvidia-ctk runtime configure --runtime=docker`; podman: `nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` |
| Hugging Face token | `export HUGGING_FACE_HUB_TOKEN=hf_...` (only for gated models) |
| Free host ports | 8000 (vLLM), 8081 (Envoy), 9002/9003/9090 (EPP), 19000 (Envoy admin) |
| Outbound network | to pull images (docker.io, ghcr.io) and model weights (Hugging Face) |
| `llmdbenchmark` CLI | `./install.sh` (Python 3.11+) |

**Not required:** Kubernetes, `kubectl`/`oc`, `helm`/`helmfile`, Gateway CRDs,
PVCs, or any cluster access.

Verify GPU-in-container before you start:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Usage

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...     # for gated models

# Bring up vLLM + EPP + Envoy as local containers (step 00 runs the preflight).
llmdbenchmark --spec config/specification/nok8s.yaml.j2 --base-dir . standup --methods nok8s

# Benchmark it — the harness runs as a local container against http://localhost:8081.
llmdbenchmark --spec config/specification/nok8s.yaml.j2 --base-dir . run

# Remove the containers.
llmdbenchmark --spec config/specification/nok8s.yaml.j2 --base-dir . teardown --methods nok8s
```

`--methods nok8s` is optional when the scenario sets `nok8s.enabled: true`
(the method is auto-detected), but harmless to pass explicitly.

Send your own requests to the Envoy front door:
```bash
curl -s http://localhost:8081/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Hello","max_tokens":32}'
```

## Configuration

Selected by `nok8s.enabled: true` in a scenario (mutually exclusive with
`modelservice`/`standalone`/`fma`/`kustomize`). See
[`config/scenarios/nok8s.yaml`](../config/scenarios/nok8s.yaml) for a complete
single-GPU example. Key fields (full defaults in
`config/templates/values/defaults.yaml`):

| Key | Meaning |
|-----|---------|
| `nok8s.runtime` | `docker` or `podman` (GPU flag switches to CDI for podman) |
| `nok8s.hfTokenEnv` | Host env var passed to vLLM for HF auth |
| `nok8s.workspaceHostDir` | Host dir where EPP/Envoy configs are staged + bind-mounted |
| `nok8s.vllm.{image,tag,hostPort,tensorParallel,gpus,shmSize,replicas,extraArgs}` | vLLM worker(s); worker *i* is published on `hostPort + i` |
| `nok8s.epp.{image,tag,grpcPort,grpcHealthPort,metricsPort}` | Endpoint Picker |
| `nok8s.envoy.{image,tag,listenPort}` | Envoy front door (the run target) |
| `model.{name,huggingfaceId}` | Set both; standup uses `name`, run reads `huggingfaceId` |

Sizing (fp16, single GPU): ~16 GB → 7–8B · ~24 GB → 8B · ~40 GB → 14B ·
~80 GB → 32B. For bigger models use `nok8s.vllm.extraArgs`
(e.g. `["--max-model-len","16384","--gpu-memory-utilization","0.95"]`) or an
FP8 checkpoint.

## How it maps to the pipeline

| Phase | nok8s behaviour |
|-------|-----------------|
| standup step 00 | Preflight: runtime / GPU / ports / token (replaces helm/kubectl checks) |
| standup steps 02–05 | Skipped (no namespace, PVC, model-download Job) |
| standup step 06 | `step_06_nok8s_deploy` launches the containers, waits for `/v1/models`, records `http://localhost:<listenPort>` |
| run step 03 | Endpoint resolves to the local Envoy URL (no cluster query) |
| run step 07 | `step_07_deploy_harness_local` runs the harness image locally with `--network host`; results land in `workspace/results/` via a bind-mount |
| teardown step 06 | `step_06_nok8s_teardown` removes the containers |

## Multiple workers

Set `nok8s.vllm.replicas: N` (needs `N` GPUs, or a small model sharing one GPU
with `--gpu-memory-utilization`). Each worker is published on `hostPort + i`
and added to the EPP endpoints file, so the router load-balances / prefix-routes
across them.

## Troubleshooting

- **Preflight fails on runtime** — install docker/podman or set `nok8s.runtime`.
- **vLLM container exits at load** — usually a too-large model for VRAM or a bad
  HF token; check `docker logs vllm-0`. Lower the model size or add
  `--max-model-len`/`--gpu-memory-utilization` via `nok8s.vllm.extraArgs`.
- **Envoy 503** — a worker isn't up; confirm `curl http://localhost:8000/v1/models`.
- **podman + GPU** — ensure the CDI spec exists: `nvidia-ctk cdi list` should show
  `nvidia.com/gpu=all`.
- **Ports busy** — a stale run; `teardown --methods nok8s`, or change the ports.
