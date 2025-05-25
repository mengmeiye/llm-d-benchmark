#!/usr/bin/env bash
source ${LLMDBENCH_CONTROL_DIR}/env.sh

announce "🛠️  Cloning and setting up fmperf locally..."
pushd ${LLMDBENCH_FMPERF_DIR} &>/dev/null
if [[ ! -d fmperf ]]; then
  llmdbench_execute_cmd "cd ${LLMDBENCH_FMPERF_DIR}; git clone \"${LLMDBENCH_FMPERF_GIT_REPO}\" -b \"${LLMDBENCH_FMPERF_GIT_BRANCH}\"" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
else
  pushd fmperf &>/dev/null
  llmdbench_execute_cmd "cd ${LLMDBENCH_FMPERF_DIR}/fmperf; git checkout ${LLMDBENCH_FMPERF_GIT_BRANCH}" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
  popd &>/dev/null
fi
pushd fmperf &>/dev/null
is_ce=$(conda env list | grep $LLMDBENCH_FMPERF_CONDA_ENV_NAME || true)
is_ce=$(echo "$is_ce" | awk '{ print $1 }')
if [[ ! -z $is_ce ]]; then
  announce "⏭️  Conda environment \"${LLMDBENCH_FMPERF_CONDA_ENV_NAME}\" already set. Skipping install."
else
  conda create -y -n "$LLMDBENCH_FMPERF_CONDA_ENV_NAME" python=3.11
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$LLMDBENCH_FMPERF_CONDA_ENV_NAME"
  pip install -r requirements.txt
  pip install -e .

  ${LLMDBENCH_CONTROL_CCMD} build -t fmperf .
  mkdir -p requests && chmod o+w requests
  cp .env.example .env
fi
popd &>/dev/null
popd &>/dev/null
announce "✅ fmperf setup locally."

announce "🔄 Creating namespace (${LLMDBENCH_FMPERF_NAMESPACE}), service account (${LLMDBENCH_FMPERF_SERVICE_ACCOUNT}) and rbac for fmperf..."
cat << EOF > $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_namespace_sa_rbac_secret.yaml
---
apiVersion: v1
kind: Namespace
metadata:
  name: ${LLMDBENCH_FMPERF_NAMESPACE}
  labels:
    kubernetes.io/metadata.name: ${LLMDBENCH_FMPERF_NAMESPACE}
$(${LLMDBENCH_CONTROL_KCMD} get namespace/${LLMDBENCH_VLLM_COMMON_NAMESPACE} -o yaml | yq .metadata.labels | grep -Ev "metadata.name" | sed 's|^|    |g')
  annotations:
$(${LLMDBENCH_CONTROL_KCMD} get namespace/${LLMDBENCH_VLLM_COMMON_NAMESPACE} -o yaml | yq .metadata.annotations | grep -Ev "last-applied|creationTimestamp" | sed 's|^|    |g')
spec:
  finalizers:
  - kubernetes
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${LLMDBENCH_FMPERF_SERVICE_ACCOUNT}
  namespace: ${LLMDBENCH_FMPERF_NAMESPACE}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: fmperf-job-creator
  namespace: ${LLMDBENCH_FMPERF_NAMESPACE}
rules:
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "get", "list", "watch", "delete", "patch", "update"]
- apiGroups: [""]
  resources: ["serviceaccounts"]
  verbs: ["get"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: fmperf-job-creator-binding
  namespace: ${LLMDBENCH_FMPERF_NAMESPACE}
subjects:
- kind: ServiceAccount
  name: fmperf-runner
  namespace: ${LLMDBENCH_FMPERF_NAMESPACE}
roleRef:
  kind: Role
  name: fmperf-job-creator
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: fmperf-restricted-scc
  namespace: ${LLMDBENCH_FMPERF_NAMESPACE}
subjects:
- kind: ServiceAccount
  name: fmperf-runner
  namespace: ${LLMDBENCH_FMPERF_NAMESPACE}
roleRef:
  kind: ClusterRole
  name: system:openshift:scc:restricted
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: v1
kind: Secret
metadata:
  name: ${LLMDBENCH_VLLM_COMMON_HF_TOKEN_NAME}
  namespace: ${LLMDBENCH_FMPERF_NAMESPACE}
data:
  HF_TOKEN: "$(echo ${LLMDBENCH_HF_TOKEN} | base64)"
EOF

llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} apply -f $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_namespace_sa_rbac_secret.yaml" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
announce "✅ Namespace (${LLMDBENCH_FMPERF_NAMESPACE}), service account (${LLMDBENCH_FMPERF_SERVICE_ACCOUNT}) and rbac for fmperf created"

for vol in ${LLMDBENCH_FMPERF_PVC_NAME}; do
  announce "🔄 Creating PVC ${vol} for fmperf data storage..."
  cat << EOF > $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_pvc_${vol}.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${vol}
  namespace: ${LLMDBENCH_FMPERF_NAMESPACE}
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

  announce "🔄 Starting pod \"access-to-fmperf-data-${vol}\" to provide access to PVC ${vol} (fmperf data storage)..."
  cat <<EOF > $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_a_pod_access_to_fmperf_data.yaml
apiVersion: v1
kind: Pod
metadata:
  name: access-to-fmperf-data-${vol}
  labels:
    app: llm-d-benchmark-fmperf
  namespace: ${LLMDBENCH_FMPERF_NAMESPACE}
spec:
  containers:
  - name: rsync
    image: ${LLMDBENCH_IMAGE_REGISTRY}/${LLMDBENCH_IMAGE_REPO}:${LLMDBENCH_IMAGE_TAG}
    imagePullPolicy: Always
    securityContext:
      runAsRoot: true
    command: ["rsync", "--daemon", "--no-detach", "--port=20873", "--log-file=/dev/stdout"]
    volumeMounts:
    - name: requests
      mountPath: /requests
  volumes:
  - name: requests
    persistentVolumeClaim:
      claimName: $LLMDBENCH_FMPERF_PVC_NAME
EOF
  llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} apply -f $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_a_pod_access_to_fmperf_data.yaml" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}

    cat << EOF > $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_b_service_access_to_fmperf_data.yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-d-benchmark-fmperf
  namespace: ${LLMDBENCH_FMPERF_NAMESPACE}
spec:
  ports:
  - name: rsync
    protocol: TCP
    port: 20873
    targetPort: 20873
  selector:
    app: llm-d-benchmark-fmperf
  type: ClusterIP
EOF

  llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} apply -f $LLMDBENCH_CONTROL_WORK_DIR/setup/yamls/${LLMDBENCH_CURRENT_STEP}_b_service_access_to_fmperf_data.yaml" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
  announce "✅ Pod \"access-to-fmperf-data-${vol}\" started, providing access to PVC ${vol}"
done

if [[ $LLMDBENCH_CONTROL_DEPLOY_IS_OPENSHIFT -eq 1 ]]; then
  llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} \
adm \
policy \
add-scc-to-user \
anyuid \
-z ${LLMDBENCH_FMPERF_SERVICE_ACCOUNT} \
-n $LLMDBENCH_FMPERF_NAMESPACE" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}

  llmdbench_execute_cmd "${LLMDBENCH_CONTROL_KCMD} \
adm \
policy \
add-scc-to-user \
privileged \
-z ${LLMDBENCH_FMPERF_SERVICE_ACCOUNT} \
-n $LLMDBENCH_FMPERF_NAMESPACE" ${LLMDBENCH_CONTROL_DRY_RUN} ${LLMDBENCH_CONTROL_VERBOSE}
fi
