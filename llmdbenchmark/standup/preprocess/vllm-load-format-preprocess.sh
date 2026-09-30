#!/usr/bin/env bash

# vLLM-ONLY. Installs the Python dependencies a vLLM `--load-format` needs and
# rewrites `--model-loader-extra-config` for it. Both are vLLM concepts --
# SGLang and TensorRT-LLM have no equivalent -- so this is opt-in: a scenario
# asks for it by name in `<role>.engine.preprocessCommand`, and nothing runs it
# by default. The GPU probe at the bottom is engine-neutral (any torch-based
# engine wants TORCH_CUDA_ARCH_LIST) and runs whenever nvidia-smi is present.
#
# Reads from the environment, all of which the scenario itself supplies via
# `<role>.extraEnvVars`:
#   LLMDBENCH_VLLM_COMMON_VLLM_LOAD_FORMAT       which loader to prepare
#   LLMDBENCH_VLLM_COMMON_MODEL_LOADER_EXTRA_CONFIG   JSON to rewrite
#   LLMDBENCH_VLLM_STANDALONE_MODEL              model id, for the tensorizer path
#   VLLM_LOGGING_LEVEL                           DEBUG selects a log config path

export LLMDBENCH_VLLM_TENSORIZER_URI=""

# export a custom log format path
shopt -s nocasematch # Enable case-insensitive matching
if [[ ${VLLM_LOGGING_LEVEL} == "DEBUG" ]]; then
    # export a custom log format path
    # the preprocess python script will create the file with custom log format
    export VLLM_LOGGING_CONFIG_PATH=/tmp/vllm_logging_config.json
fi
shopt -u nocasematch # Disable case-insensitive matching

# unescape double quotes if existent
export LLMDBENCH_VLLM_COMMON_MODEL_LOADER_EXTRA_CONFIG=$(echo "$LLMDBENCH_VLLM_COMMON_MODEL_LOADER_EXTRA_CONFIG" | sed 's/\\"/"/g')

# installs dependencies for load formats
if [[ ${LLMDBENCH_VLLM_COMMON_VLLM_LOAD_FORMAT} == "fastsafetensors" ]]; then
    pip install --root-user-action=ignore fastsafetensors==0.1.15
elif [[ ${LLMDBENCH_VLLM_COMMON_VLLM_LOAD_FORMAT} == "tensorizer" ]]; then
    sudo apt update
    sudo apt install -y jq
    pip install --root-user-action=ignore tensorizer==2.12.0
    # path to save serialized file
    export LLMDBENCH_VLLM_TENSORIZER_URI="/tmp/${LLMDBENCH_VLLM_STANDALONE_MODEL}/v1/model.tensors"
    export LLMDBENCH_VLLM_COMMON_MODEL_LOADER_EXTRA_CONFIG=$(echo "$LLMDBENCH_VLLM_COMMON_MODEL_LOADER_EXTRA_CONFIG" | jq '.tensorizer_uri = env.LLMDBENCH_VLLM_TENSORIZER_URI' | tr -d '\n')
elif [[ ${LLMDBENCH_VLLM_COMMON_VLLM_LOAD_FORMAT} == "runai_streamer" ]]; then
    sudo apt update
    sudo apt install -y jq
    pip install --root-user-action=ignore runai==0.4.1
    # controls the level of concurrency and number of OS threads
    # reading tensors from the file to the CPU buffer
    # https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md
    export LLMDBENCH_VLLM_COMMON_MODEL_LOADER_EXTRA_CONFIG=$(echo "$LLMDBENCH_VLLM_COMMON_MODEL_LOADER_EXTRA_CONFIG" | jq '.concurrency = 32' | tr -d '\n')
fi

echo "vllm extra arguments: '${LLMDBENCH_VLLM_COMMON_MODEL_LOADER_EXTRA_CONFIG}'"

# Sets TORCH_CUDA_ARCH_LIST from the GPUs this pod can actually see.
#
# This used to select which of nvidia-smi's rows to trust by matching them
# against a GPU name parsed out of LLMDBENCH_VLLM_COMMON_AFFINITY, an env var
# the standalone template rendered as "nvidia.com/gpu:<labelKey>:<labelValue>".
# Two things were wrong with that. It was a node label standing in for what the
# pod was granted -- but the kubelet has already decided that by the time this
# runs, so every row nvidia-smi prints IS a device this pod will use, and there
# is nothing to filter. And the match was between a label value
# ("NVIDIA-H100-80GB-HBM3") and a product name with spaces swapped for hyphens,
# which silently found nothing the moment a cluster labelled its nodes
# differently. nvidia-smi already reports the name, so nothing is lost.
if command -v nvidia-smi >/dev/null 2>&1; then
    compute_cap_list=""
    declare -A compute_cap_map
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        fullname=$(echo $line | cut -d ',' -f 1 | xargs)
        compute_cap=$(echo $line | cut -d ',' -f 2 | xargs)
        [[ -n "$compute_cap" ]] || continue
        # add compute capability if not added already
        if [[ ! -n "${compute_cap_map[$compute_cap]}" ]]; then
            compute_cap_map[$compute_cap]=1
            compute_cap_list="${compute_cap_list:+${compute_cap_list};}$compute_cap"
        fi
        uuid=$(echo $line | cut -d ',' -f 3 | xargs)
        persistence_mode=$(echo $line | cut -d ',' -f 4 | xargs)
        echo "gpu_uuid='$uuid' gpu_name='$fullname' compute_cap='$compute_cap' persistence_mode='$persistence_mode'"
    done < <( nvidia-smi --query-gpu=name,compute_cap,uuid,persistence_mode --format=csv,noheader,nounits )

    if [[ -n "$compute_cap_list" ]]; then
        export TORCH_CUDA_ARCH_LIST=$compute_cap_list
        echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
    fi
fi
