## Concept
Use a specific harness to generate workloads against a stack serving a large language model, according to a specific workload profile. To this end, a new `pod`, `llmdbench-${LLMDBENCH_HARNESS_NAME}-launcher`, is created on the target cluster, with an associated `pvc` (by default `workload-pvc`) to store experimental data. Once the "launcher" `pod` completes its run - which will include data collection **and data analysis** - the experimental data is then extracted from the "workload-pvc" back to the experimenter's workstation.

> [!NOTE]
> The first `run` against a namespace prepares the harness infrastructure: the harness namespace, an HF token secret, and a preprocess ConfigMap, plus (unless `--no-pvc` is set) the workload PVC and data-access pod. On a fresh namespace this first run pays the PVC bind wait (`--pvc-bind-timeout`, default 240s) before the harness pod can start; later runs against the same namespace reuse the existing infrastructure.

## Metrics
For a discussion of candidate relevant metrics, please consult this [document](https://docs.google.com/document/d/1SpSp1E6moa4HSrJnS4x3NpLuj88sMXr2tbofKlzTZpk/edit?resourcekey=0-ob5dR-AJxLQ5SvPlA4rdsg&tab=t.0#heading=h.qmzyorj64um1)

| Category | Metric | Unit |
| ---------| ------- | ----- |
| Throughput | Output tokens / second | tokens / second |
| Throughput | Input tokens / second | tokens / second |
| Throughput | Requests / second | qps |
| Latency    | Time per output token (TPOT) | ms per output token |
| Latency    | Time to first token (TTFT) | ms |
| Latency    | Time per request (TTFT + TPOT * output length) | seconds per request |
| Latency    | Normalized time per output token (TTFT/output length +TPOT) aka NTPOT | ms per output token |
| Latency    | Inter Token Latency (ITL) - Time between decode tokens within a request | ms per output token |
| Correctness | Failure rate | queries |
| Experiment | Benchmark duration | seconds |

## Workloads
For a discussion of relevant workloads, please consult this [document](https://docs.google.com/document/d/1Ia0oRGnkPS8anB4g-_XPGnxfmOTOeqjJNb32Hlo_Tp0/edit?tab=t.0)

| Workload                               | Use Case            | ISL    | ISV   | OSL    | OSV    | OSP    | Latency   |
| -------------------------------------- | ------------------- | ------ | ----- | ------ | ------ | ------ | ----------|
| Interactive Chat                       | Chat agent          | Medium | High  | Medium | Medium | Medium | Per token |
| Classification of text                 | Sentiment analysis  | Medium |       | Short  | Low    | High   | Request   |
| Classification of images               | Nudity filter       | Long   | Low   | Short  | Low    | High   | Request   |
| Summarization / Information Retrieval  | Q&A from docs, RAG  | Long   | High  | Short  | Medium | Medium | Per token |
| Text generation                        |                     | Short  | High  | Long   | Medium | Low    | Per token |
| Translation                            |                     | Medium | High  | Medium | Medium | High   | Per token |
| Code completion                        | Type ahead          | Long   | High  | Short  | Medium | Medium | Request   |
| Code generation                        | Adding a feature    | Long   | High  | Medium | High   | Medium | Request   |

## Profiles
A list of pre-defined profiles, each specific to particular harness, can be found on subdirectories under `workloads/profiles`.

```
📦 workload
 + 📂 profiles
 | + 📂 guidellm
 | | + 📜 sanity_concurrent.yaml.in
 | + 📂 nop
 | | + 📜 nop.yaml.in
 | + 📂 inference-perf
 | | + 📜 sanity_random.yaml.in
 | | + 📜 summarization_synthetic.yaml.in
 | | + 📜 chatbot_sharegpt.yaml.in
 | | + 📜 shared_prefix_synthetic.yaml.in
 | | + 📜 chatbot_synthetic.yaml.in
 | | + 📜 code_completion_synthetic.yaml.in
 | + 📂 vllm-benchmark
 | | + 📜 sanity_random.yaml.in
 | | + 📜 random_concurrent.yaml.in
```
What is shown here are the workload profile **templates** (hence, the `yaml.in`) and for each template, parameters which are specific for a particular standup are automatically replaced to generate a `yaml`. This rendered workload profile is then stored as a `configmap` on the target `Kubernetes` cluster. An illustrative example follows (`inference-perf/sanity_random.yaml.in`) :

```
load:
  type: constant
  stages:
  - rate: 1
    duration: 30
api:
  type: completion
  streaming: true
server:
  type: vllm
  model_name: REPLACE_ENV_LLMDBENCH_DEPLOY_CURRENT_MODEL
  base_url: REPLACE_ENV_LLMDBENCH_HARNESS_STACK_ENDPOINT_URL
  ignore_eos: true
tokenizer:
  pretrained_model_name_or_path: REPLACE_ENV_LLMDBENCH_DEPLOY_CURRENT_MODEL
data:
  type: random
  input_distribution:
    min: 10             # min length of the synthetic prompts
    max: 100            # max length of the synthetic prompts
    mean: 50            # mean length of the synthetic prompts
    std_dev: 10         # standard deviation of the length of the synthetic prompts
    total_count: 100    # total number of prompts to generate to fit the above mentioned distribution constraints
  output_distribution:
    min: 10             # min length of the output to be generated
    max: 100            # max length of the output to be generated
    mean: 50            # mean length of the output to be generated
    std_dev: 10         # standard deviation of the length of the output to be generated
    total_count: 100    # total number of output lengths to generate to fit the above mentioned distribution constraints
report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: true
storage:
  local_storage:
    path: /workspace
```

Entries `REPLACE_ENV_LLMDBENCH_DEPLOY_CURRENT_MODEL` and `REPLACE_ENV_LLMDBENCH_HARNESS_STACK_ENDPOINT_URL` will be automatically replaced with the current value of the environment variables `LLMDBENCH_DEPLOY_CURRENT_MODEL` and `LLMDBENCH_HARNESS_STACK_ENDPOINT_URL` respectively.

In addition to that, **any other parameter on the workload profile can be overwritten** by passing a list of `<key>=<value>` pairs to `-o/--overrides`.

Finally, new workload profiles can be hand-written and placed under the correct directory. Once written, they are selected with `-w/--workload`.

## Use
`llmdbenchmark run` with no parameters beyond the scenario uses the defaults below.

If a stack was stood up from a highly customized scenario (a different model, a specific context length, a specific network card), pass the same scenario to `run` so the harness targets it correctly:

```bash
llmdbenchmark --spec guides/optimized-baseline run -l inference-perf -w sanity_random -o min=20,total_count=200
```

Command line parameters override individual entries in the workload profile, as `-o` does above.

> [!IMPORTANT]
> `run` can, and usually is, used against a stack that was deployed by other means -- outside `llmdbenchmark standup`. Point it at the endpoint with `-U/--endpoint-url`.

Everything below has both a command line flag and an environment-variable form, `LLMDBENCH_` plus the flag's long name, upper-cased.

### Choosing the load

| Flag | Environment variable | Meaning |
| ---- | -------------------- | ------- |
| `-l`/`--harness` | `LLMDBENCH_HARNESS` | Harness (load generator) to run. Default `inference-perf`; also settable as `harness.name`. |
| `-w`/`--workload` | `LLMDBENCH_WORKLOAD` | Workload profile the harness runs. Default `sanity_random.yaml`; also settable as `harness.experimentProfile`. A bare name resolves inside `workload/profiles/<harness name>`. |
| `--workload-file-path` | `LLMDBENCH_WORKLOAD_FILE_PATH` | Point at a profile outside the profiles tree. |
| `-o`/`--overrides` | `LLMDBENCH_OVERRIDES` | Comma-separated `key=value` pairs overriding entries in the workload profile. |
| `-e`/`--experiments` | `LLMDBENCH_EXPERIMENTS` | Sweep definition (see [doe.md](doe.md)). |
| `-x`/`--dataset` | `LLMDBENCH_DATASET` | Dataset to fetch into the harness pod, for profiles that replay one. |
| `-j`/`--parallelism` | `LLMDBENCH_PARALLELISM` | How many harness pods generate load, all running the same profile. Default `1`; also `harness.loadParallelism`. |

### Choosing the target

| Flag | Environment variable | Meaning |
| ---- | -------------------- | ------- |
| `-p`/`--namespace` | `LLMDBENCH_NAMESPACE` | Namespace the stack was stood up in, and where the harness pod is created. |
| `-m`/`--model` | `LLMDBENCH_MODEL` | Which model of the stack to run against. |
| `--stack` | `LLMDBENCH_STACK` | Which named stacks of a multi-stack scenario to run. Default: every stack. |
| `-t`/`--methods` | `LLMDBENCH_METHODS` | Standup methods the stack used, so `run` knows where to look for it. |
| `-U`/`--endpoint-url` | `LLMDBENCH_ENDPOINT_URL` | Target an endpoint directly instead of discovering it from the stack. |
| `--list-endpoints` | -- | Print the endpoints `run` would discover, and exit. |

### Harness pod and results

| Flag | Environment variable | Meaning | Scenario key |
| ---- | -------------------- | ------- | ------------ |
| `--wait-timeout` | `LLMDBENCH_WAIT_TIMEOUT` | How long to wait for the harness pod to finish. Default `3600`. | `harness.waitTimeout` |
| `-r`/`--output` | `LLMDBENCH_OUTPUT` | Where results land. Default `local`. | `harness.output` |
| `--data-collect` | `LLMDBENCH_DATA_COLLECT` | How much result data is copied to this machine: `default` (`oc cp`), `fast` (gzip'd `oc exec \| tar`), `results` (reports/metadata/plots only) or `skip` (nothing; results stay on the PVC). | -- |
| `--compress` / `--no-compress` | `LLMDBENCH_COMPRESS` | Compress each result set on the PVC before collecting it, so the archive rather than the raw tree crosses the tunnel. Benchmark reports, `run_metadata.yaml`, `experiment-summary.yaml` and plots stay plain. Default on. | -- |
| `--compress-level` | `LLMDBENCH_COMPRESS_LEVEL` | zstd compression level. Default `10`. | -- |
| `--no-pvc` | `LLMDBENCH_NO_PVC` | Run without the workload PVC and data-access pod: harness pods use an emptyDir and results are copied straight out of them. | -- |
| `-z`/`--skip` | `LLMDBENCH_SKIP` | Skip execution and only collect data already on the PVC. | -- |
| `-d`/`--debug` | `LLMDBENCH_DEBUG` | Run the harness pod in debug mode (`sleep infinity`) so you can exec into it. | `harness.debug` |
| `-g`/`--envvarspod` | `LLMDBENCH_HARNESS_ENVVARS_TO_YAML` | Extra environment variables to add to every harness pod. | -- |
| `-q`/`--serviceaccount` | `LLMDBENCH_SERVICE_ACCOUNT` | ServiceAccount for the harness pod. | `serviceAccount.name` |
| `--monitoring` | `LLMDBENCH_MONITORING` | Collect engine metrics alongside the run (see [metrics_collection.md](metrics_collection.md)). | `monitoring.metricsScrapeEnabled` |

CPU, memory and PVC size for the harness pod are scenario keys rather than flags: `harness.resources.cpu`, `harness.resources.memory`, `harness.resources.memoryLimit`, `harness.pvcSize` and `storage.workloadPvc.name`. `llmdbenchmark run --help` is the authoritative flag list.

## Multi-Stack Runs

When a scenario defines more than one stack (e.g.
[`examples/multi-model-optimized-baseline`](../config/scenarios/examples/multi-model-optimized-baseline.yaml)),
every per-stack step in the `run` phase executes once per rendered stack -
endpoint detection, model verification, profile rendering, configmap creation,
harness deploy, wait, and collect. Each stack's results are collected into
its own experiment-ID-keyed subdirectory under the workspace. For
copy-paste recipes covering the whole multi-model lifecycle, see
[multi-model.md](multi-model.md).

**Per-stack endpoints.** For shared-HTTPRoute scenarios (`httpRoute.mode: shared`
in the scenario file), step 03 `detect_endpoint` bakes the stack's path prefix
into the detected URL - e.g. `http://gw:80/qwen3-06b` for stack `qwen3-06b`.
Every downstream step treats the endpoint as an opaque base URL, so:

- `test_model_serving` hits `http://gw:80/qwen3-06b/v1/models`.
- The harness pod env var `LLMDBENCH_HARNESS_STACK_ENDPOINT_URL` becomes
  `http://gw:80/qwen3-06b`; the harness script then calls
  `${endpoint_url}/v1/completions` which resolves to
  `http://gw:80/qwen3-06b/v1/completions`.
- The shared HTTPRoute rewrites `/qwen3-06b/*` -> `/*` so the upstream vLLM
  still sees `/v1/completions` and its friends.

Nothing in the harness scripts changes - the routing prefix is invisible to them.

**Parallelism knobs.** `--parallel N` (default 4) caps how many stacks the
executor runs per-stack steps for at once. Set `--parallel 1` to serialize
for easier debugging, especially when multi-stack harness pods compete for
the same accelerator nodes. (Note: the `smoketest` phase always runs stacks
sequentially regardless of `--parallel`, since interleaved `/health` and
`/v1/models` probes across stacks make failures harder to read.)

### Discovering deployed endpoints

After standup, `--list-endpoints` prints a table of per-stack routing URLs
and a copy-paste block of ready-to-run invocations - no harness pods
launched:

```bash
llmdbenchmark --spec examples/multi-model-optimized-baseline run -p <namespace> --list-endpoints
```

Useful when you've forgotten the stack names, the gateway IP, or just want
a quick sanity-check that both pools resolved correctly. The flag runs
the full render pipeline (so the printed endpoints match exactly what a
real `standup` would produce) and then short-circuits before launching
any harness pods.

### Targeting a single pool (`--stack`)

`--stack NAME` (or comma-separated list) restricts run execution to one
stack. Endpoint URL auto-resolves for the selected stack - no need to
pass `--endpoint-url` manually:

```bash
# Benchmark qwen3-06b only with guidellm, two parallel harness pods
llmdbenchmark --spec examples/multi-model-optimized-baseline run -p <namespace> \
  --stack qwen3-06b \
  -l guidellm -w sanity_random.yaml -j 2
```

`--stack` also works on `standup`, `smoketest`, and `teardown`. Unknown
names fail loudly with a list of valid ones. Available via
`LLMDBENCH_STACK` env var too.

### CLI overrides in multi-stack scenarios

| Flag | Multi-stack behavior |
|------|----------------------|
| `-p / --namespace` | Applies to every stack (namespaces are scenario-wide). |
| `-t / --methods` | Applies to every stack. |
| `--monitoring` | Applies to every stack. |
| `-u / --wva` | Applies to every stack. |
| `-l / --harness`, `-w / --workload`, `-o / --overrides` | Applies to every stack's harness pod - all stacks run the same workload. |
| `-j / --parallelism` | Applies to every stack - each stack launches N parallel harness pods. |
| `--endpoint-url` | Single endpoint for run-only mode; bypasses auto-detect. In multi-stack, only meaningful when combined with `--stack` (otherwise every stack targets the same endpoint, which is rarely desired). |
| `--stack NAME[,NAME...]` | Scopes every per-stack step to the named subset. |
| **`--set KEY=VALUE`** (scenario overrides) | **Applies to every stack unless the key carries a `stack:` or glob prefix** - e.g. `--set 'llama-31-8b:decode.replicas=4'`. Unlike `-m`, an unprefixed `--set` still applies to every stack when `--stack` narrows the deployment: `--stack` picks what is *deployed*, the prefix picks what is *modified*. A selector matching no stack is a hard error. See [standup.md](standup.md#scoping-overrides-in-multi-stack-scenarios). |
| **`-m / --models`** | **Scopes to the filter when `--stack NAME` names exactly one stack** - only that stack's model is overridden; siblings keep their scenario-defined models. Without `--stack` (or with a broader filter), `-m` applies to every stack and emits a warning - that collapses a multi-model scenario into N copies of one model, which is almost never desired. |

So the clean pattern for "rerun pool A against a different model":

```bash
llmdbenchmark --spec examples/multi-model-optimized-baseline run -p <ns> \
  --stack qwen3-06b \
  --model meta-llama/Llama-3.2-3B \
  -l inference-perf -w sanity_random.yaml
```

The filter scopes `-m` to one stack; sibling stacks are left alone.

The rule: anything that's inherently per-stack configuration (model name,
path, shortName) is best edited in the scenario YAML, not overridden via
CLI. CLI flags are designed to override *scenario-wide* knobs (namespace,
harness, workload) uniformly across every stack - or, when combined with
`--stack`, to target a single stack without disturbing the others.

### Benchmarking a single stack from a multi-stack scenario

Preferred - use `--stack`, endpoint auto-resolves:

```bash
# After standup of examples/multi-model-optimized-baseline
llmdbenchmark --spec examples/multi-model-optimized-baseline run -p <namespace> \
  --stack qwen3-06b \
  -l inference-perf -w sanity_random.yaml
```

Equivalent - pin `--endpoint-url` yourself (useful if the scenario file
isn't available locally):

```bash
llmdbenchmark run \
  --endpoint-url http://<gateway>:80/qwen3-06b \
  --model Qwen/Qwen3-0.6B \
  --namespace <namespace> \
  -l inference-perf -w sanity_random.yaml
```

Include the path prefix in the URL exactly as shown - the HTTPRoute
rewrites it away before the request reaches vLLM.

## Harnesses

### [inference-perf](https://github.com/kubernetes-sigs/inference-perf)

### [guidellm](https://github.com/vllm-project/guidellm.git)

### [vLLM benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks)

### Nop (No Op)

The `nop` harness runs no load. Against a `standalone` stack it parses the engine log and reports weight-loading time statistics instead.

What it needs from the scenario, all under `standalone`:

| Scenario key | Value | Why |
| ------------ | ----- | --- |
| `standalone.engine.command` | include `--load-format <format>` -- `safetensors`, `tensorizer`, `runai_streamer`, `fastsafetensors` | The loader whose timing you are measuring. It is an engine flag, so it lives in the command. |
| `standalone.engine.command` | include `--enable-sleep-mode` | Required for sleep/wake benchmarks. |
| `standalone.extraEnvVars` | `VLLM_LOGGING_LEVEL: DEBUG` | An engine environment variable, not a flag. At `DEBUG` the preprocess script points vLLM at a log format the `nop` categories report can parse; at anything less, categories go missing. |
| `standalone.engine.preprocessCommand` | `source /setup/preprocess/vllm-load-format-preprocess.sh ; /setup/preprocess/vllm-load-format-preprocess.py` | Installs the loader's dependencies, exports what it needs, and pre-serializes the model for the `tensorizer` format. Runs in the standalone pod, in the engine's shell, before the command. |
| `harness.name` | `nop` | |

A loader that the harness also has to know about -- `tensorizer`, for instance -- reads its format from `LLMDBENCH_VLLM_COMMON_VLLM_LOAD_FORMAT` in `standalone.extraEnvVars`; set it there and reference it in the command as `--load-format $LLMDBENCH_VLLM_COMMON_VLLM_LOAD_FORMAT` so the pod and the harness see one value.

#### With the FMA launcher

A second container can be added to a `standalone` stack that runs the inference launcher from [llm-d-fast-model-actuation](https://github.com/llm-d-incubation/llm-d-fast-model-actuation/blob/main/inference_server/launcher/launcher.py). Its image also contains vLLM, and that image is used for both containers so they run under identical conditions.

| Scenario key | Default | Meaning |
| ------------ | ------- | ------- |
| `standalone.launcher.enabled` | `false` | Add the launcher container. |
| `standalone.launcher.port` | `8001` | Port the launcher listens on. |
| `standalone.launcher.vllmPort` | `8002` | Port the vLLM server it starts waits on. |
| `standalone.launcher.image.repository` / `.tag` | -- | The launcher image. |
| `standalone.launcher.customPreprocessCommand` | -- | Preprocess command for the launcher container. |

With the launcher on, the `nop` harness reports metrics for both the standalone server and the launched one. [`examples/launcher.yaml`](../config/scenarios/examples/launcher.yaml) is a worked example.
