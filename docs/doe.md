## Concept
A `yaml` file which contains a list of `standup` and `run` parameters of interest, termed `factors` and a list of values of interest, termed `levels` for each one of the `factors`. The each set of values for each factor produces a list of combinations termed `treatments`. These concepts and nomenclature follow the "Design of Experiments" (DOE) approach, and it allows a systematic and reproducible investigation on how different parameters affect the overall performance of a stack.

## Motivation
While the triplet `<scenario>`,`<harness>`,`<(workload) profile>`, contains all information required for the `llm-d-benchmark` to be able to carry out a `standup`->`run`->`teardown` [lifecycle](lifecycle.md), in order to compare and validate the performance of different stacks, a large number of parameters on `llm-d` must be swept. Hence, the need for an automated mechanism to loop through this (potentially) large parameter space.

## Use
An experiment file has to be manually crafted as a `yaml`, placed under the `experiments` folder, and handed to the `llmdbenchmark experiment` command.

> [!NOTE]
> `llmdbenchmark experiment` (which **combines** `llmdbenchmark standup`, `llmdbenchmark run` and `llmdbenchmark teardown`) is the only command that can have an experiment file supplied to it.

| Flag               | Environment variable    | Meaning                                          |
| ------------------ | ----------------------- | ------------------------------------------------ |
| `-e/--experiments` | `LLMDBENCH_EXPERIMENTS` | `yaml` file containing an experiment description |

> [!TIP]
> A name without a leading directory resolves inside the `experiments` folder, and the `.yaml` suffix is optional -- `--experiments pd-disaggregation` is enough.

> [!NOTE]
> For detailed information on the experiment lifecycle, including how treatments are executed and results are collected, see [llmdbenchmark/experiment/README.md](../llmdbenchmark/experiment/README.md).

## Anatomy of an experiment file

| Block              | Read by                | Contents                                                                                                |
| ------------------ | ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `experiment`       | reporting              | `name`, `description`, and the `harness` / `profile` the design was written for.                          |
| `design`           | nobody -- documentation | The factor space: `setup.factors` and `run.factors`, each a list of `{name, key, levels, description}`, plus totals and response variables. This is the record of *why* the treatments below look the way they do. |
| `setup.treatments` | the orchestrator       | The infrastructure configurations to sweep. Each is a named map of dotted **scenario** paths, merged into the scenario before standup. |
| `setup.constants`  | the orchestrator       | Scenario overrides held fixed across every treatment.                                                    |
| `treatments`       | the orchestrator       | The load configurations to sweep. Each is a named map of **workload profile** fields, applied against a deployed stack without redeploying it. |

A setup key is resolved exactly like a `--set` key, so anything addressable in
a scenario is sweepable: `decode.replicas`, `prefill.parallelism.tensor`,
`standalone.enabled`, `router.epp.pluginsConfigFile`, `model.maxModelLen`.

## Illustrative examples

1) Compare `standalone` vllm with `llm-d` in a stack with a variable number of `prefill` and `decode` `pods`. Each time a new combination is deployed, run a workload profile with varying `max-concurrecy` and `num-prompts`

> [!IMPORTANT]
> The harness - `vllm-benchmark` and (workload) `profile` (`random_concurrent`) are **not** defined here, but on the [scenario](standup.md#scenarios)

```yaml
experiment:
  name: pd-disaggregation
  harness: vllm-benchmark
  profile: random_concurrent.yaml

design:
  type: fractional_factorial
  setup:
    factors:
      - name: deploy_method
        key: deploy_method
        levels: [modelservice, standalone]
        description: Deployment method
      - name: decode_replicas
        key: decode.replicas
        levels: [1, 2, 3, 4]
      - name: decode_tensor_parallelism
        key: decode.parallelism.tensor
        levels: [2, 4, 8]
      - name: prefill_replicas
        key: prefill.replicas
        levels: [2, 4, 6, 8]
      - name: prefill_tensor_parallelism
        key: prefill.parallelism.tensor
        levels: [1, 2, 4]
  run:
    factors:
      - name: max-concurrency
        key: max-concurrency
        levels: [1, 8, 32, 64, 128, 256]
      - name: num-prompts
        key: num-prompts
        levels: [10, 80, 320, 640, 1280, 2560]

setup:
  treatments:
    # Each of these sweeps the chart-side width only, for brevity. A real spec
    # restates `<role>.engine.command` at the same width alongside it -- see the
    # note below.
    - name: ms-d1xTP8-p8xTP1
      decode.replicas: 1
      decode.parallelism.tensor: 8
      prefill.replicas: 8
      prefill.parallelism.tensor: 1

    - name: ms-d2xTP4-p4xTP2
      decode.replicas: 2
      decode.parallelism.tensor: 4
      prefill.replicas: 4
      prefill.parallelism.tensor: 2

    # A standalone treatment has to turn modelservice off explicitly: the
    # scenario's own value survives the merge otherwise, and both deploy
    # methods would contend for the same model PVC.
    - name: sa-d1xTP8
      standalone.enabled: true
      modelservice.enabled: false
      decode.replicas: 1
      decode.parallelism.tensor: 8

treatments:
  - name: conc1
    max-concurrency: 1
    num-prompts: 10

  - name: conc64
    max-concurrency: 64
    num-prompts: 640

  - name: conc256
    max-concurrency: 256
    num-prompts: 2560
```

> [!NOTE]
> A treatment names only the keys it changes -- there is no placeholder for a key it does not use. `decode.replicas` in the `sa-` treatments above sizes the standalone deployment; the `prefill.*` keys are simply absent there, because a standalone stack has no prefill role.

The parallelism widths are the one case worth a second look, because a width is
two facts, not one. `decode.parallelism.tensor` is the chart value that sizes the
pod's accelerator request; the width the engine itself runs at is a flag inside
`decode.engine.command`. A treatment that sweeps one must sweep the other, or the
pod is granted eight devices and the engine shards across two.

A treatment **replaces** the command string rather than patching it, so restate
the whole line -- connector configuration included -- at the new width. See
[developer-guide.md](developer-guide.md#custom-doe-specifications) for a spec
written that way, and [standup.md](standup.md#what-comes-from-the-scenario-file)
for which facts come from the command and which are stated.

** This particular example can be used with the following command :

```
llmdbenchmark --spec guides/pd-disaggregation experiment --experiments pd-disaggregation
```

2) Compare different endpoint-picker (EPP) routing plugin configurations, using a fixed set of `decode` `pods`. Once deployed, run a workload profile varying `num_groups` and `system_prompt_len`)

> [!IMPORTANT]
> The harness - `inference-perf` and (workload) `profile` (`shared_prefix_synthetic`) are **not** defined here, but on the [scenario](standup.md)

```yaml
experiment:
  name: precise-prefix-cache-aware
  harness: inference-perf
  profile: shared_prefix_synthetic.yaml

design:
  type: full_factorial
  setup:
    factors:
      - name: pluginsConfigFile
        key: router.epp.pluginsConfigFile
        levels: [default-plugins.yaml, prefix-cache-estimate-config.yaml, prefix-cache-tracking-config.yaml]
        description: Endpoint-picker plugin configuration
  run:
    factors:
      - name: num_groups
        key: data.shared_prefix.num_groups
        levels: [40, 60]
      - name: system_prompt_len
        key: data.shared_prefix.system_prompt_len
        levels: [8000, 5000, 1000]

setup:
  constants:
    model.maxModelLen: 16000
    model.blockSize: 64
  treatments:
    - name: routing-default
      router.epp.pluginsConfigFile: default-plugins.yaml
    - name: routing-estimate
      router.epp.pluginsConfigFile: prefix-cache-estimate-config.yaml
    - name: routing-tracking
      router.epp.pluginsConfigFile: prefix-cache-tracking-config.yaml

treatments:
  - name: grp40-splen8k
    data.shared_prefix.num_groups: 40
    data.shared_prefix.system_prompt_len: 8000
  - name: grp60-splen5k
    data.shared_prefix.num_groups: 60
    data.shared_prefix.system_prompt_len: 5000
  - name: grp60-splen1k
    data.shared_prefix.num_groups: 60
    data.shared_prefix.system_prompt_len: 1000
```

> [!NOTE]
> `router.epp.pluginsConfigFile` has to be written at that level. One level up, under `router`, it parses and reaches the chart values where nothing reads it, and the endpoint picker runs its default configuration instead.

** This particular example can be used with the following command

```
llmdbenchmark --spec guides/precise-prefix-cache-routing experiment --experiments precise-prefix-cache-aware
```


## Treatment Execution Lifecycle

The `experiment` command orchestrates a nested loop of setup and run treatments. Understanding this lifecycle is important for designing experiments and interpreting results.

### Setup Treatment Cycling

Each **setup treatment** represents a distinct infrastructure configuration. The experiment command cycles through setup treatments sequentially, performing the full standup/run/teardown lifecycle for each:

```
For each setup treatment:
    1. standup   -- Deploy the stack with this treatment's infrastructure parameters
    2. run       -- Execute ALL run treatments against this stack
    3. teardown  -- Tear down the stack before moving to the next setup treatment
```

For example, if you have 3 setup treatments (different replica counts) and 5 run treatments (different concurrency levels), the experiment executes:

```
Setup treatment 1 (replicas=2):
    standup to run treatment 1..5 to teardown
Setup treatment 2 (replicas=4):
    standup to run treatment 1..5 to teardown
Setup treatment 3 (replicas=8):
    standup to run treatment 1..5 to teardown
```

This produces 15 result sets total (3 setup x 5 run treatments).

### Setup Treatments vs `--set`

`setup.treatments` and the `--set` CLI flag write to the same place: dotted
paths deep-merged into each stack's config before rendering. The difference
is intent -- a treatment is a **factor being swept** (it varies per cycle and
is recorded in the experiment summary), while `--set` is a **constant for the
whole invocation**.

When both touch the same key, the treatment wins, so a sweep is never
silently flattened by a CLI flag. Use `--set` for things that hold across
every cycle (a cluster's storage class, a backend flip) and `setup.treatments`
for the variable under study.

The `stack:` / glob selector prefix is currently a `--set` feature only;
`setup.treatments` keys apply to every stack. See
[standup.md](standup.md#scoping-overrides-in-multi-stack-scenarios).

### Run Treatments Within Each Cycle

Within a single setup treatment, run treatments are executed sequentially against the same deployed stack. Each run treatment applies its factor values as workload profile overrides (e.g., `max-concurrency=64, num-prompts=640`). The harness pod is deployed, executes the workload, collects results, and is cleaned up before the next run treatment begins.

### Step Ordering

During each lifecycle phase, steps execute in three partitions:

1. **Pre-global steps** -- Global steps that run before any per-stack work (e.g., preflight checks, cleanup of previous runs)
2. **Per-stack steps** -- Steps that operate on each stack individually (e.g., endpoint detection, profile rendering, harness deployment, result collection)
3. **Post-global steps** -- Global steps that run after all per-stack work completes (e.g., result upload, post-run cleanup, local analysis)

For the run phase specifically, steps 00-01 are pre-global, steps 02-08 are per-stack, and steps 09-11 are post-global. This ensures that operations like result upload and analysis happen only after all per-stack collection is complete.

## Parallelism Levels

The benchmark supports parallelism at three levels:

| Level | Flag | Description |
|-------|------|-------------|
| **Pod parallelism** | `-j/--parallelism` | Number of identical harness pods running the same workload profile concurrently. Useful for increasing aggregate load or reducing statistical variance. Each pod writes results to its own subdirectory (`<experiment_id>_1`, `<experiment_id>_2`, etc.). |
| **Treatment parallelism** | (sequential) | Run treatments within a setup cycle execute sequentially. Each treatment waits for the previous to complete before starting. |
| **Setup parallelism** | `--parallel N` | Maximum number of stacks to deploy in parallel during standalone `standup` (not used during `experiment`, which processes setup treatments sequentially to avoid resource contention). |

Pod parallelism is the most commonly used level. With `-j 4`, four harness pods are created simultaneously, each executing the same workload profile. Results from all pods are collected and can be analyzed together for statistical significance.
