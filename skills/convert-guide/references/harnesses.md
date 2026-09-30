# Harnesses and Workload Profiles

The harness is the load generator that runs *against* the stack you converted.
It is not in the guide -- the guide deploys the server. Take it from the user's
request, or use the defaults.

```yaml
    harness:
      name: inference-perf
      experimentProfile: shared_prefix_synthetic.yaml
```

Defaults if the user says nothing: `inference-perf` / `sanity_random.yaml`.

## Harnesses

One directory per harness under `workload/profiles/`, one launcher per harness
under `workload/harnesses/`:

| `harness.name` | What it runs |
|---|---|
| `inference-perf` | Default. Kubernetes-native load generator; the widest profile set. |
| `guidellm` | Alternative load generator (concurrency sweeps, multi-turn). |
| `vllm-benchmark` | vLLM's own `benchmark_serving`. Works against any OpenAI-compatible endpoint, not just vLLM. |
| `aiperf` | NVIDIA AIPerf: synthetic ISL/OSL and dataset/trace replay. |
| `inferencemax` | InferenceMAX-style saturation sweep. |
| `lm-eval` | Accuracy evaluation (`lm-evaluation-harness`), not throughput. |
| `eval-containers` | Agentic evaluations that run in their own container (GAIA, Aider Polyglot). |
| `priority-mix` | Mixed-priority request stream, for scheduling and fairness work. |
| `nop` | Stands the stack up and runs no load. Use it when the conversion itself is what you are testing. |

## Profiles

Profiles live at `workload/profiles/<harness>/<profile>.yaml.in`. On disk they
carry `.in` because they are templates rendered with the run's values;
**reference them without it** -- `experimentProfile: sanity_random.yaml` loads
`sanity_random.yaml.in`.

Do not trust a list in this file to stay current. Read the directory:

```bash
ls workload/profiles/inference-perf/
```

The names that recur across scenarios:

| Profile | Use |
|---|---|
| `sanity_random.yaml` | Smallest thing that proves the stack serves. Default. |
| `random_concurrent.yaml` | Concurrency sweep on random prompts. |
| `shared_prefix_synthetic.yaml` | Prefix-cache and cache-routing work (`_short`, `_heavy` variants exist). |
| `chatbot_synthetic.yaml`, `chatbot_sharegpt.yaml` | Chat shapes, synthetic and replayed. |
| `summarization_synthetic.yaml` | Long input, short output. |
| `code_completion_synthetic.yaml`, `agentic_code_generation.yaml` | Code shapes. |
| `interactive-chat.yaml` | Latency-sensitive interactive traffic. |
| `otel_traces.yaml`, `weka_traces.yaml` | Trace replay. |
| `nop.yaml` | The `nop` harness's only profile. |

`guide_<guide-name>_<n>.yaml` profiles are per-guide load definitions committed
alongside a converted scenario. If the guide you are converting specifies its
own load (a `benchmark` step, a `vllm bench` invocation in the README), add one
of these rather than bending a generic profile -- copy the closest
`guide_*.yaml.in` as a starting point and name it after your guide.

`aiperf` dataset replay needs the dataset fetched into the harness pod: pass
`--dataset s3://...` or set `experiment.datasetUrl`.
