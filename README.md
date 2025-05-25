## `llm-d`-benchmark

This repository provides an automated workflow for benchmarking LLM inference using the `llm-d` stack. It includes tools for deployment, experiment execution, data collection, and teardown across multiple environments and deployment styles.

### Goal

To provide a single source of automation for repeatable and reproducible experiments and performance evaluation on `llm-d`.

### Architecture

The benchmarking system drives synthetic or trace-based traffic into an llm-d-powered inference stack, orchestrated via Kubernetes. Requests are routed through a scalable load generator, with results collected and visualized for latency, throughput, and cache effectiveness.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)">
    <img alt="llm-d Logo" src="./docs/images/llm-d-benchmarking.jpg" width=100%>
  </picture>
</p>


### Main concepts (identified by specific directories)

#### Scenarios

Pieces of information identifying a particular cluster. This inforation includes, but it is not limited to, GPU model, llm model and llm-d parameters (an environment file, and optionally a "values.yaml" file for llm-d-deployer)

#### Harness

Load Generator (python code), written using software facilites available at https://github.com/fmperf-project/fmperf.

> [!NOTE]
> This will be expanded with additional load generators in the future.

#### Workload

FMPerf workload specification, with load profile (e.g., `share-gpt` vs `long-input`) and load levels (e.g., QPS values). IMPORTANT: these definitions will be expanded with specifications for other load generators,

> [!IMPORTANT]
> The triple `<scenario>`,`<harness>`,`<workload>`, combined with the standup/teardown capabilities provided by llm-d-deployer (https://github.com/llm-d/llm-d-deployer) should provide enough information to allow an experiment to be reproduced.

### Dependecies:
- llm-d-deployer (https://github.com/llm-d/llm-d-deployer)
- fm-perf: https://github.com/fmperf-project/fmperf

### 📦 Repository Setup
```
git clone https://github.com/llm-d/llm-d-benchmark.git
cd llm-d-benchmark
```

## Quickstart

#### Standing up llm-d for experimentation and benchmarking
```
export LLMDBENCH_CLUSTER_HOST="https://api.fmaas-platform-eval.fmaas.res.ibm.com"
export LLMDBENCH_CLUSTER_TOKEN="..."
```
> [!TIP]
> You can simply use your current context. **After running kubectl/oc login**, just set `export LLMDBENCH_CLUSTER_HOST=auto` (and leave LLMDBENCH_CLUSTER_TOKEN unconfigured)

> [!IMPORTANT]
> No matter which method used (i.e., fully specify `LLMDBENCH_CLUSTER_HOST` and `LLMDBENCH_CLUSTER_TOKEN` or simply use the current context, there is an additional variable which will always require definition: `LLMDBENCH_HF_TOKEN`

A complete list of available variables (and its default values) can be found by running
 `cat setup/env.sh | grep "^export LLMDBENCH_" | sort`

> [!NOTE]
> The `namespaces` specified by the environment variables `LLMDBENCH_VLLM_COMMON_NAMESPACE` and `LLMDBENCH_FMPERF_SERVICE_ACCOUNT` will be automatically created.

> [!TIP]
> If you want all generated `yaml` files and all data collected to reside on the same directory, set the environment variable `LLMDBENCH_CONTROL_WORK_DIR` explicitly before starting execution.

#### List of "standup steps"
Run the command line with the option `-h` in order to produce a list of steps
```
./setup/standup.sh -h
```
> [!NOTE]
> Each individual "step file" is name in a way that briefly describes each one the multiple steps required for a full deployment.

> [!TIP]
> Steps 0-5 can be considered "preparation" and can be skipped in most deployments.

#### to dry-run
```
./setup/standup.sh -n
```

### Deployment

vLLM instances can be deployed by one of the following methods:
- "standalone" (a simple deployment with services associated to the deployment)
- "deployer" (invoking \"llm-d-deployer\").

This is controlled by the environment variable LLMDBENCH_DEPLOY_METHODS (default "deployer"). The value of the environment variable can be overriden by the paraemeter `-t/--types` (applicable for both `teardown.sh` and `standup.sh`)

> [!WARNING]
> At this time, only **one simultaneous** deployment method is supported

All available models are listed and controlled by the variable `LLMDBENCH_DEPLOY_MODEL_LIST`. The value of the above mentioned environment variable can be overriden by the paraemeter `-m/--model` (applicable for both `teardown.sh` and `standup.sh`).

> [!WARNING]
> At this time, only **one simultaneous** model is supported

> [!TIP]
> The following aliases can be used in place of the full mode name, for convenience (_llama-3b_ -> `meta-llama/Llama-3.2-3B-Instruct`, _llama-8b_ -> `meta-llama/Llama-3.1-8B-Instruct`, _llama-70b_ -> `meta-llama/Llama-3.1-70B-Instruct`, _llama-17b_ -> `RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic`)

### Scenarios

All relevant variables to a particular experiment are stored in a "scenario" (folder aptly named).

The expectation is that an experiment is run by initially executing:

```
source scenario/<scenario name>
```

### Lifecycle (Standup/Run/Teardown)

At this point, with all the environment variables set (tip, `env | grep ^LLMDBENCH_ | sort`) you should be ready to deploy and test
```
./setup/standup.sh
```

> [!NOTE]
> The scenario can also be indicated as part of the command line optios for `standup.sh` (e.g. `./setup/standup.sh -c ocp_H100MIG_deployer_llama-3b`)

To re-execute only individual steps (full name or number):
```
./setup/standup.sh --step 08_smoketest.sh
./setup/standup.sh -s 7
./setup/standup.sh -s 3-5
./setup/standup.sh -s 5,7
```

Once llm-d is fully deployed, an experiment can be run
```
./run.sh
```
> [!IMPORTANT]
> This command will run an experiment, collect data and perform an initial analysis (generating statistics and plots). One can go straight to the analysis by adding the option `-z`/`--skip` to the above command

> [!NOTE]
> The scenario can also be indicated as part of the command line optios for `run.sh` (e.g., `./run.sh -c ocp_L40_standalone_llama-8b`)

Finally, cleanup everything
```
./setup/teardown.sh
```
> [!NOTE]
> The scenario can also be indicated as part of the command line optios for `teardown.sh` (e.g., `./teardown.sh -c kubernetes_H200_deployer_llama-8b`)

## Quickstart K8s Launcher with Analysis

For a simplified workflow that includes automated analysis of benchmark results, check out the quickstart-k8s launcher. This workflow provides:

- Easy deployment and execution of benchmarks on Kubernetes
- Support for comparing multiple LLM models
- Generation of comprehensive performance visualizations

### Quickstart Workflows

1. **Single Model Benchmark**: Run benchmarks for a single model with automated analysis
   - See [Single Model Quickstart](quickstart-k8s/README.md) for details

2. **Multi-Model Comparison**: Compare performance across multiple LLM models
   - See [Multi-Model Comparison Quickstart](quickstart-k8s/Compare-README.md) for details

To get started, navigate to the quickstart-k8s directory and follow the instructions in the respective README files.