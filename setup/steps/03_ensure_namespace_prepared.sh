#!/usr/bin/env bash
source ${LLMDBENCH_CONTROL_DIR}/env.sh

announce "🔍 Checking if \"${LLMDBENCH_CLUSTER_NAMESPACE}\" is prepared."
call_prepare=0
is_ns=$(${LLMDBENCH_CONTROL_KCMD} get namespace | grep ${LLMDBENCH_CLUSTER_NAMESPACE} || true)
if [[ -z ${is_ns} ]]; then
  call_prepare=1
fi

is_secret=$(${LLMDBENCH_CONTROL_KCMD} --namespace ${LLMDBENCH_CLUSTER_NAMESPACE} get --no-headers secret ${LLMDBENCH_VLLM_COMMON_HF_TOKEN_NAME} --ignore-not-found=true)
if [[ -z ${is_secret} ]]; then
  call_prepare=1
fi

is_pvc=$(${LLMDBENCH_CONTROL_KCMD} --namespace ${LLMDBENCH_CLUSTER_NAMESPACE} get --no-headers pvc $LLMDBENCH_VLLM_COMMON_PVC_NAME --ignore-not-found=true)
if [[ -z ${is_pvc} ]]; then
  call_prepare=1
fi

for model in ${LLMDBENCH_DEPLOY_MODEL_LIST//,/ }; do
  cat << EOF > $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_prepare_namespace_${model}.yaml
sampleApplication:
  enabled: true
  baseConfigMapRefName: basic-gpu-with-nixl-and-redis-lookup-preset
  model:
    modelArtifactURI: pvc://$LLMDBENCH_VLLM_COMMON_PVC_NAME/models/$(model_attribute $model model)
    modelName: "$(model_attribute $model model)"
EOF

llmd_opts="--skip-infra --namespace ${LLMDBENCH_CLUSTER_NAMESPACE} --storage-class ${LLMDBENCH_VLLM_COMMON_PVC_STORAGE_CLASS} --storage-size ${LLMDBENCH_VLLM_COMMON_PVC_MODEL_CACHE_SIZE} --values-file $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_prepare_namespace_${model}.yaml"
announce "🚀 Calling llm-d-deployer with options \"${llmd_opts}\"..."
llmdbench_execute_cmd "cd $LLMDBENCH_DEPLOYER_DIR/llm-d-deployer/quickstart; export KUBECONFIG=$LLMDBENCH_CONTROL_WORK_DIR/environment/context.ctx; export HF_TOKEN=$LLMDBENCH_HF_TOKEN; export PREPARE_ONLY=true; ./llmd-installer.sh --namespace ${LLMDBENCH_CLUSTER_NAMESPACE} $llmd_opts" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE} 0
announce "✅ llm-d-deployer prepared namespace"
done

for vol in ${LLMDBENCH_FMPERF_PVC_NAME}; do
  announce "📜 Creating PVC ${vol} for fmperf data storage..."
  cat << EOF > $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_pvc_${vol}.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${vol}
  namespace: ${LLMDBENCH_CLUSTER_NAMESPACE}
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: ${LLMDBENCH_FMPERF_PVC_SIZE}
  storageClassName: ${LLMDBENCH_VLLM_COMMON_PVC_STORAGE_CLASS}
EOF
  llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} apply -f $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_pvc_${vol}.yaml" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
  announce "✅ PVC ${vol} for fmperf data storage created"
done

if [[ $LLMDBENCH_CONTROL_DEPLOY_IS_OPENSHIFT -eq 1 ]]; then
  llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} \
adm \
policy \
add-scc-to-user \
anyuid \
-z ${LLMDBENCH_CLUSTER_SERVICE_ACCOUNT} \
-n $LLMDBENCH_CLUSTER_NAMESPACE" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}

  llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} \
adm \
policy \
add-scc-to-user \
privileged \
-z ${LLMDBENCH_CLUSTER_SERVICE_ACCOUNT} \
-n $LLMDBENCH_CLUSTER_NAMESPACE" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
fi
announce "✅ Namespace \"${LLMDBENCH_CLUSTER_NAMESPACE}\" prepared."
