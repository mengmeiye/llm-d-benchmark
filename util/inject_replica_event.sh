#!/usr/bin/env bash
# Inject one elastic event into a running benchmark: delete a model-server
# pod (failover) or change the replica count (scale-out / scale-in), after a
# delay, and record when it happened so the run's time-series metrics can be
# aligned to it.
#
# Usage:
#   inject_replica_event.sh -n NAMESPACE -e fail|scale [-d DELAY_S]
#       [-l LABEL_SELECTOR] [-r REPLICAS] [-o EVENTS_FILE] [-f]
#
#   -e fail    Delete one ready model-server pod (the Deployment recreates it).
#              -f force-deletes it with no grace period, so in-flight requests
#              fail as they would on a crash.
#   -e scale   Scale the model-server Deployment to -r REPLICAS.
#   -d         Seconds to wait before the event (default 120), so the run
#              reaches steady state first.
#   -l         Model-server pod selector (default llm-d.ai/role=decode).
#   -o         Append a JSON line describing the event to this file.

set -euo pipefail

NAMESPACE=""
EVENT=""
DELAY_S=120
SELECTOR="llm-d.ai/role=decode"
REPLICAS=""
EVENTS_FILE=""
FORCE=false

while getopts "n:e:d:l:r:o:fh" opt; do
  case "${opt}" in
    n) NAMESPACE="${OPTARG}" ;;
    e) EVENT="${OPTARG}" ;;
    d) DELAY_S="${OPTARG}" ;;
    l) SELECTOR="${OPTARG}" ;;
    r) REPLICAS="${OPTARG}" ;;
    o) EVENTS_FILE="${OPTARG}" ;;
    f) FORCE=true ;;
    h | *) sed -n '2,20p' "$0"; exit 0 ;;
  esac
done

if [[ -z "${NAMESPACE}" || -z "${EVENT}" ]]; then
  echo "error: -n NAMESPACE and -e fail|scale are required" >&2
  exit 2
fi
if [[ "${EVENT}" == "scale" && -z "${REPLICAS}" ]]; then
  echo "error: -e scale needs -r REPLICAS" >&2
  exit 2
fi

record() {
  local detail="$1"
  local line
  line=$(printf '{"time":"%s","epoch":%s,"namespace":"%s","event":"%s",%s}' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(date +%s)" "${NAMESPACE}" "${EVENT}" "${detail}")
  echo "${line}"
  if [[ -n "${EVENTS_FILE}" ]]; then
    echo "${line}" >> "${EVENTS_FILE}"
  fi
}

echo "waiting ${DELAY_S}s before the ${EVENT} event"
sleep "${DELAY_S}"

case "${EVENT}" in
  fail)
    POD=$(kubectl get pods -n "${NAMESPACE}" -l "${SELECTOR}" \
      -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name} {.spec.nodeName}{"\n"}{end}' \
      | head -1)
    if [[ -z "${POD}" ]]; then
      echo "error: no running pod matches ${SELECTOR} in ${NAMESPACE}" >&2
      exit 1
    fi
    read -r POD_NAME NODE <<< "${POD}"
    if [[ "${FORCE}" == true ]]; then
      kubectl delete pod -n "${NAMESPACE}" "${POD_NAME}" --grace-period=0 --force --wait=false
    else
      kubectl delete pod -n "${NAMESPACE}" "${POD_NAME}" --wait=false
    fi
    record "$(printf '"pod":"%s","node":"%s","force":%s' "${POD_NAME}" "${NODE}" "${FORCE}")"
    ;;
  scale)
    DEPLOYMENT=$(kubectl get deployments -n "${NAMESPACE}" -l "${SELECTOR}" -o jsonpath='{.items[0].metadata.name}')
    if [[ -z "${DEPLOYMENT}" ]]; then
      echo "error: no Deployment matches ${SELECTOR} in ${NAMESPACE}" >&2
      exit 1
    fi
    FROM=$(kubectl get deployment -n "${NAMESPACE}" "${DEPLOYMENT}" -o jsonpath='{.spec.replicas}')
    kubectl scale deployment -n "${NAMESPACE}" "${DEPLOYMENT}" --replicas="${REPLICAS}"
    record "$(printf '"deployment":"%s","from":%s,"to":%s' "${DEPLOYMENT}" "${FROM}" "${REPLICAS}")"
    ;;
  *)
    echo "error: unknown event ${EVENT} (want fail or scale)" >&2
    exit 2
    ;;
esac
