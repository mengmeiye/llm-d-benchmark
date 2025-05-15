#!/usr/bin/env bash
source ${LLMDBENCH_CONTROL_DIR}/env.sh

if [[ $LLMDBENCH_USER_IS_ADMIN -eq 1 ]]; then
  announce "Setting up inference-gateway using KGateway..."
  if [[ $(${LLMDBENCH_CONTROL_KCMD} get pods -n kgateway-system --no-headers --ignore-not-found  --field-selector status.phase=Running | wc -l) -ne 0 ]]; then
    echo "❗ KGateway already installed."
  else
    if [[ $LMDBENCH_CONTROL_ENVIRONMENT_TYPE_P2P_ACTIVE -eq 1 ]]; then
      pushd ${LLMDBENCH_GAIE_DIR} &>/dev/null
      if [[ ! -d llm-d-inference-scheduler ]]; then
          llmdbench_execute_cmd "git clone https://github.com/llm-d/llm-d-inference-scheduler/" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
      fi
      pushd llm-d-inference-scheduler &>/dev/null
      if [[ $LLMDBENCH_CONTROL_DEPLOY_IS_OPENSHIFT -eq 1 ]]; then
        llmdbench_execute_cmd "make install-openshift" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
      else
        llmdbench_execute_cmd "make install-k8s" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
      fi
      popd &>/dev/null
      popd &>/dev/null
    else
      announce "ℹ️ Environment types are \"${LLMDBENCH_DEPLOY_METHODS}\". Skipping this step."
    fi
  fi

  _wiev1=$(${LLMDBENCH_CONTROL_KCMD} get crd -o "custom-columns=NAME:.metadata.name,VERSIONS:spec.versions[*].name" | grep -E "workload.*istio.*v1," || true)
  if [[ -z ${_wiev1} ]]; then
    announce "Installing the latest CRDs from istio..."
    llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} apply -f https://raw.githubusercontent.com/istio/istio/refs/tags/1.23.1/manifests/charts/base/crds/crd-all.gen.yaml" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE} 0 3
  else
    announce "ℹ️ Latest CRDs from istio already installed"
  fi
else
    announce "❗No privileges to setup KGateway. Will assume an user with proper privileges already performed this action."
fi