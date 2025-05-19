#!/usr/bin/env bash

# Copyright 2025 The llm-d Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

if [[ $0 != "-bash" ]]; then
    pushd `dirname "$(realpath $0)"` > /dev/null 2>&1
fi

export LLMDBENCH_BASE64_CONTEXT=${LLMDBENCH_BASE64_CONTEXT:-}
if [[ ! -z $LLMDBENCH_BASE64_CONTEXT ]]; then
  echo ${LLMDBENCH_BASE64_CONTEXT} | base64 -d > ~/.kube/$LLMDBENCH_CONTROL_REMOTE_KUBECONFIG_FILENAME
fi

export LLMDBENCH_ENV_VAR_LIST=$(env | grep ^LLMDBENCH | cut -d '=' -f 1)
export LLMDBENCH_CONTROL_DIR=$(realpath $(pwd)/)

if [ $0 != "-bash" ] ; then
    popd  > /dev/null 2>&1
fi

export LLMDBENCH_MAIN_DIR=$(realpath ${LLMDBENCH_CONTROL_DIR}/../)

source ${LLMDBENCH_CONTROL_DIR}/env.sh

export LLMDBENCH_CONTROL_DRY_RUN=${LLMDBENCH_CONTROL_DRY_RUN:-0}
export LLMDBENCH_CONTROL_VERBOSE=${LLMDBENCH_CONTROL_VERBOSE:-0}
export LLMDBENCH_DEPLOY_SCENARIO=
export LLMDBENCH_CLIOVERRIDE_DEPLOY_SCENARIO=
export LLMDBENCH_FMPERF_EXPERIMENT_SKIP=0

function show_usage {
    echo -e "Usage: $(echo $0 | rev | cut -d '/' -f 1 | rev) -n/--dry-run [just print the command which would have been executed (default=$LLMDBENCH_CONTROL_DRY_RUN) ] \n \
             -c/--scenario [take environment variables from a scenario file (default=$LLMDBENCH_DEPLOY_SCENARIO) ] \n \
             -m/--models [list the models to be run against (default=$LLMDBENCH_DEPLOY_MODEL_LIST) ] \n \
             -z/--skip [skip the execution of the experiment, and only collect data (default=$LLMDBENCH_FMPERF_EXPERIMENT_SKIP) ] \n \
             -v/--verbose [print the command being executed, and result (default=$LLMDBENCH_CONTROL_VERBOSE) ] \n \
             -h/--help (show this help)"
}

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -c=*|--scenario=*)
        export LLMDBENCH_CLIOVERRIDE_DEPLOY_SCENARIO=$(echo $key | cut -d '=' -f 2)
        ;;
        -c|--scenario)
        export LLMDBENCH_CLIOVERRIDE_DEPLOY_SCENARIO="$2"
        shift
        ;;
        -n|--dry-run)
        export LLMDBENCH_CLIOVERRIDE_CONTROL_DRY_RUN=1
        export LLMDBENCH_ENV_VAR_LIST=$LLMDBENCH_ENV_VAR_LIST" LLMDBENCH_CONTROL_DRY_RUN"
        ;;
        -m=*|--models=*)
        export LLMDBENCH_CLIOVERRIDE_DEPLOY_MODEL_LIST=$(echo $key | cut -d '=' -f 2)
        export LLMDBENCH_ENV_VAR_LIST=$LLMDBENCH_ENV_VAR_LIST" LLMDBENCH_DEPLOY_MODEL_LIST"
        ;;
        -m|--models)
        export LLMDBENCH_CLIOVERRIDE_DEPLOY_MODEL_LIST="$2"
        export LLMDBENCH_ENV_VAR_LIST=$LLMDBENCH_ENV_VAR_LIST" LLMDBENCH_DEPLOY_MODEL_LIST"
        shift
        ;;
        -z|--skip)
        export LLMDBENCH_CLIOVERRIDE_FMPERF_EXPERIMENT_SKIP=1
        ;;
        -v|--verbose)
        export LLMDBENCH_CLIOVERRIDE_CONTROL_VERBOSE=1
        export LLMDBENCH_ENV_VAR_LIST=$LLMDBENCH_ENV_VAR_LIST" LLMDBENCH_CONTROL_VERBOSE"
        ;;
        -h|--help)
        show_usage
        if [[ "${BASH_SOURCE[0]}" == "${0}" ]]
        then
            exit 0
        else
            return 0
        fi
        ;;
        *)
        echo "ERROR: unknown option \"$key\""
        show_usage
        exit 1
        ;;
        esac
        shift
done

export LLMDBENCH_CONTROL_CLI_OPTS_PROCESSED=1

source ${LLMDBENCH_CONTROL_DIR}/env.sh

export LLMDBENCH_EXPERIMENT_ID=${LLMDBENCH_EXPERIMENT_ID:-"default"}

if [[ $LLMDBENCH_FMPERF_REMOTE_EXECUTION -eq 1 ]]; then
  announce "🚀 Running experiment (harness=$LLMDBENCH_FMPERF_EXPERIMENT_HARNESS, profile=$LLMDBENCH_FMPERF_EXPERIMENT_PROFILE) remotely ..."

  export LLMDBENCH_BASE64_CONTEXT=$(cat $LLMDBENCH_CONTROL_WORK_DIR/environment/context.ctx | base64)

  env_vars_cmd_cli_opts="--env LLMDBENCH_FMPERF_REMOTE_EXECUTION=0 --env LLMDBENCH_FMPERF_DIR=/workspace/fmperf"
  for i in ${LLMDBENCH_ENV_VAR_LIST} LLMDBENCH_BASE64_CONTEXT LLMDBENCH_CONTROL_REMOTE_KUBECONFIG_FILENAME; do
    if [[ $i != "LLMDBENCH_FMPERF_REMOTE_EXECUTION" ]]; then
      echo $i
      env_vars_cmd_cli_opts=" --env=\"$i=${!i}\" $env_vars_cmd_cli_opts"
    fi
  done

  llmdbench_run_cli_opts=""
  if [[ ! -z $LLMDBENCH_DEPLOY_SCENARIO ]]; then
    llmdbench_run_cli_opts=$llmdbench_run_cli_opts" -c $LLMDBENCH_DEPLOY_SCENARIO"
  fi

  if [[ $LLMDBENCH_FMPERF_EXPERIMENT_SKIP -ne 0 ]]; then
    llmdbench_run_cli_opts=$llmdbench_run_cli_opts" -z"
  fi
  announce "⏳ Waiting for pod \"fmperfrunpod\" to complete its execution..."
  llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} --namespace $LLMDBENCH_CLUSTER_NAMESPACE run fmperfrunpod ${env_vars_cmd_cli_opts} -i --image-pull-policy Always --attach --pod-running-timeout ${LLMDBENCH_CONTROL_WAIT_TIMEOUT}s --restart=Never --rm --image=icr.io/vopo/llm-d-benchmark:latest --command -- bash -c \"./llm-d-benchmark/run.sh $llmdbench_run_cli_opts -n\""  ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE} 0
  announce "✅ Experiment completed successfully"

else

  announce "🚀 Running experiment (harness=$LLMDBENCH_FMPERF_EXPERIMENT_HARNESS, profile=$LLMDBENCH_FMPERF_EXPERIMENT_PROFILE) locally ..."

  pushd ${LLMDBENCH_FMPERF_DIR}/fmperf &>/dev/null

# Hardcode Conda init from known working path

  if [ "$LLMDBENCH_CONTROL_DEPLOY_HOST_OS" = "mac" ] && [ -f "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" ]; then
    llmdbench_execute_cmd "source \"/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh\"" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
  elif [ "$LLMDBENCH_CONTROL_DEPLOY_HOST_OS" = "linux" ] && [ -f "/opt/miniconda/etc/profile.d/conda.sh" ]; then
    llmdbench_execute_cmd "source \"/opt/miniconda/etc/profile.d/conda.sh\"" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
  else
    echo "❌ Could not find conda.sh for $LLMDBENCH_CONTROL_DEPLOY_HOST_OS. Please verify your Anaconda installation."
    exit 1
  fi
  llmdbench_execute_cmd "conda activate \"$LLMDBENCH_FMPERF_CONDA_ENV_NAME\"" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}

  if [[ ${LLMDBENCH_CONTROL_DRY_RUN} -eq 0 ]]; then
# Confirm we're using the correct Python environment
    announce "✅ Python: $(which $LLMDBENCH_CONTROL_PCMD)"
    announce "✅ Env: $(conda info --envs | grep '*' || true)"
    ${LLMDBENCH_CONTROL_PCMD} -m pip show urllib3 >/dev/null 2>&1 || ${LLMDBENCH_CONTROL_PCMD} -m pip install urllib3
    ${LLMDBENCH_CONTROL_PCMD} -m pip show kubernetes >/dev/null 2>&1 || ${LLMDBENCH_CONTROL_PCMD} -m pip install kubernetes
    ${LLMDBENCH_CONTROL_PCMD} -m pip show pandas >/dev/null 2>&1 || ${LLMDBENCH_CONTROL_PCMD} -m pip install pandas
    pip install -e . >/dev/null 2>&1
  fi

  llmdbench_execute_cmd "cp -f ${LLMDBENCH_MAIN_DIR}/workload/harnesses/$LLMDBENCH_FMPERF_EXPERIMENT_HARNESS ${LLMDBENCH_CONTROL_WORK_DIR}/workload/harnesses/$LLMDBENCH_FMPERF_EXPERIMENT_HARNESS" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}

  for model in ${LLMDBENCH_DEPLOY_MODEL_LIST//,/ }; do
    export LLMDBENCH_DEPLOY_CURRENT_MODEL=$model
    render_template ${LLMDBENCH_MAIN_DIR}/workload/profiles/$LLMDBENCH_FMPERF_EXPERIMENT_PROFILE.in ${LLMDBENCH_CONTROL_WORK_DIR}/workload/profiles/$LLMDBENCH_FMPERF_EXPERIMENT_PROFILE
    unset LLMDBENCH_DEPLOY_CURRENT_MODEL
  done

  if [[ $LLMDBENCH_CONTROL_ENVIRONMENT_TYPE_STANDALONE_ACTIVE -eq 1 ]]; then
    export LLMDBENCH_FMPERF_STACK_TYPE=vllm-prod
    export LLMDBENCH_FMPERF_SERVICE_URL=$(${LLMDBENCH_CONTROL_KCMD} --namespace "$LLMDBENCH_CLUSTER_NAMESPACE" get service --no-headers | grep standalone | awk '{print $1}' || true)
  else
    export LLMDBENCH_FMPERF_STACK_TYPE=llm-d
    export LLMDBENCH_FMPERF_SERVICE_URL=$(${LLMDBENCH_CONTROL_KCMD} --namespace "$LLMDBENCH_CLUSTER_NAMESPACE" get gateway --no-headers | tail -n1 | awk '{print $1}')
  fi

  ecmd="${LLMDBENCH_CONTROL_PCMD} ${LLMDBENCH_CONTROL_WORK_DIR}/workload/harnesses/$LLMDBENCH_FMPERF_EXPERIMENT_HARNESS"
  if [[ $LLMDBENCH_FMPERF_EXPERIMENT_SKIP -eq 0 ]]; then
    announce "⏳ Starting the actual execution ..."
    if [[ ${LLMDBENCH_CONTROL_DRY_RUN} -eq 0 ]]; then
      $ecmd
    else
      echo "---> would have executed the command \"$ecmd\""
    fi
  else
    announce "⏭️ Skipping experiment execution"
  fi
  announce "✅ Actual execution completed successfully"

  llmdbench_execute_cmd "touch $(pwd)/pod_log_response.txt; mv -f $(pwd)/pod_log_response.txt ${LLMDBENCH_CONTROL_WORK_DIR}/results/pod_log_response.txt" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}

  popd &>/dev/null
fi

announce "🏗️ Collecting results ..."
PN=$(echo $LLMDBENCH_EXPERIMENT_ID | $LLMDBENCH_CONTROL_SCMD 's^_^-^g' | tr '[:upper:]' '[:lower:]')

  cat <<EOF > $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/run_access_to_pvc.yaml
apiVersion: v1
kind: Pod
metadata:
  name: $PN
  namespace: $LLMDBENCH_CLUSTER_NAMESPACE
spec:
  containers:
  - name: rsync
    image: busybox
    command: ["sleep", "infinity"]
    volumeMounts:
    - name: requests
      mountPath: /requests
  volumes:
  - name: requests
    persistentVolumeClaim:
      claimName: $LLMDBENCH_FMPERF_PVC_NAME
EOF

llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} apply -f $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/run_access_to_pvc.yaml" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} --namespace ${LLMDBENCH_CLUSTER_NAMESPACE} wait --timeout=${LLMDBENCH_CONTROL_WAIT_TIMEOUT}s --for=condition=Ready=True pod/$PN" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} --namespace ${LLMDBENCH_CLUSTER_NAMESPACE} cp $PN:/requests/${LLMDBENCH_EXPERIMENT_ID}/ ${LLMDBENCH_CONTROL_WORK_DIR}/results/" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} --namespace ${LLMDBENCH_CLUSTER_NAMESPACE} delete pod $PN" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
announce "✅ All results collected successfully"