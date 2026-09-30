#!/usr/bin/env bash
set -o pipefail

# -----------------------------------------------------------------------
# test-scenarios.sh
#
# This script is for TESTING ONLY. It validates that scenario configs
# produce working deployments -- either by rendering them (no cluster) or by
# running standup/teardown cycles against a live cluster. It does NOT run
# benchmarks or collect results.
#
# What to test is selected from the scenarios themselves, not from lists kept
# here: util/scenario-inventory.py reads every file under config/scenarios/ and
# reports which engine each role launches and which deploy methods the scenario
# supports. So a new scenario, or a new engine, is picked up with no edit here.
#
# ONE RUN PER NAMESPACE. Runs are sequential by design -- one scenario is stood
# up, validated and torn down before the next begins -- and two runs must not
# share a namespace, whatever they deploy:
#   * Names the upstream charts fix. With `gateway.className: epponly` the
#     router chart renders a ConfigMap named literally `envoy`, so the second
#     release in a namespace never installs: helm refuses it as "invalid
#     ownership metadata". This bites two *different* scenarios too.
#   * Names derived from the scenario. A Deployment's name comes from the stack
#     name and the model's shortName, so two runs of one scenario render the
#     same name: the second rollout replaces the first, and the loser's
#     `validate_config` then inspects a pod built from the winner's plan and
#     fails on checks its own plan got right.
# Neither failure reads as what it is. `preflight_ns` below refuses to start a
# standup in a namespace that already holds a deployment -- including one still
# downloading weights, which is most of a standup's wall clock -- so pass
# --allow-busy-ns if you have a reason, or use a second namespace.
#
# SEQUENTIAL RUNS SHARE THE MODEL PVC. One thing a standup deliberately leaves
# behind is the model PVC, so the next scenario does not download the weights
# again -- and a scenario that wants a bigger one than the last then cannot
# start, because a bound PVC cannot be grown. `prepare_model_pvcs` below deletes
# an undersized one before the standup (--keep-pvcs to stop it).
#
# See `./util/test-scenarios.sh --help`.
# -----------------------------------------------------------------------

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INVENTORY="${REPO}/util/scenario-inventory.py"

# Prefer the repo's venv: the inventory imports llmdbenchmark.engine.
PYTHON="${REPO}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

usage() {
  cat <<'EOF'
Usage: ./util/test-scenarios.sh [options] [namespace]

Validates scenarios by rendering them, or by standing them up and tearing them
down again. What runs is read from config/scenarios/ (see
util/scenario-inventory.py), so nothing here goes stale when scenarios change.

MODE
  --plan            render only -- no cluster, ~2s per scenario. Catches
                    template, schema and engine-command errors. Chart versions
                    stay "auto" in this mode, so the post-render
                    `helmfile template` check cannot run: it logs an
                    "improper constraint: auto" error that is not a scenario
                    failure. Use --standup to exercise the charts.
  --standup         standup, then teardown (the default). Needs a cluster.
  --list            print what would run, then exit.

WHAT TO RUN (combine freely; with none of these, the curated default set runs)
  --engine NAME     scenarios that run this engine (repeatable): vllm, sglang,
                    trtllm, sim. A scenario that launches NAME runs as it is;
                    one that carries NAME as a commented-out alternative
                    (tagged `# @engine NAME`) is run with `llmdbenchmark
                    --engine NAME`, which switches it in. The switch is written
                    into the workspace -- config/scenarios/ is never edited.
  --method NAME     run with `-t NAME` (repeatable): modelservice, standalone,
                    kustomize, nok8s, fma. Without this, each scenario runs the
                    method(s) it enables itself.
  --group DIR       scenarios under this directory: examples, guides, cicd,
                    experimental.
  --accel cpu       only scenarios that request no accelerator (cpu, sim ...).
  --spec SPEC       one scenario, e.g. guides/pd-disaggregation (repeatable).
                    Named scenarios are never skipped.
  --all             every scenario the inventory lists.
  --skip SPEC       drop a scenario (repeatable).
  --no-skip         also run the scenarios skipped by default (see SKIPPED).

PASS-THROUGH
  --kustomize-backend gpu/sglang
                    set kustomize.acceleratorBackend, i.e. pick the engine the
                    upstream guide is deployed with. Implies -t kustomize.
  --set KEY=VALUE   forwarded to llmdbenchmark (repeatable).
  --workspace DIR   where plans/logs go (default: /tmp/llmdbench-test-<stamp>).

TESTING A DIFFERENT ENGINE
  Under the modelservice method the engine comes from the scenario's own
  `<role>.engine.command`, and a scenario can supply one two ways: live, or as a
  commented-out group tagged `# @engine NAME` for the reader to switch in.
  `--engine NAME` covers both:

    ./util/test-scenarios.sh --plan --engine sglang
    ./util/test-scenarios.sh --plan --engine trtllm

  A scenario that launches NAME runs unchanged. A scenario that offers NAME
  (examples/engines and guides/optimized-baseline both do, for SGLang and
  TRT-LLM) is run with `llmdbenchmark --engine NAME`, which writes the switched
  scenario under the workspace and renders that -- so the commented block is
  held to the same standard as a live one, and config/scenarios/ is never
  edited. The run listing says which scenarios were switched; the ALT column of
  `util/scenario-inventory.py` is where that comes from.

  Under the kustomize method the upstream llm-d guide supplies the command and
  the scenario's own command is never read, so there is nothing to switch:
  `kustomize.acceleratorBackend` picks the engine instead, and `--engine` is
  routed to it. Any kustomize-capable guide can therefore run with any engine:

    ./util/test-scenarios.sh --method kustomize --engine sglang
    # equivalently: --method kustomize --kustomize-backend gpu/sglang

EXAMPLES
  ./util/test-scenarios.sh --plan --all --no-skip     # render everything, no cluster
  ./util/test-scenarios.sh --list --all               # what would run, with engines
  ./util/test-scenarios.sh llm-d-my-ns                # default set on a cluster
  ./util/test-scenarios.sh --accel cpu llm-d-my-ns    # no GPU needed
  ./util/test-scenarios.sh --spec guides/pd-disaggregation --method modelservice ns
  ./util/test-scenarios.sh --engine sim --plan        # the simulator paths
  ./util/test-scenarios.sh --plan --engine trtllm     # incl. commented-in TRT-LLM

  --allow-busy-ns   stand up even though the namespace is already in use. Off
                    by default: a second deployment there either fails its
                    router install outright (the `epponly` chart fixes one
                    ConfigMap name, so helm refuses the second release) or
                    replaces the first run's pods and fails validation on a plan
                    that was correct. Use a second namespace where you can.

  --keep-pvcs       do not delete a model PVC that is smaller than the next
                    scenario needs. A model PVC survives teardown so the weights
                    do not download twice, so a 300Gi one left by an earlier
                    scenario stops a standup that wants 1Ti, and it cannot be
                    grown. By default that PVC is DELETED before the standup
                    (printed when it happens) and the weights download again.
                    With this flag the standup fails on the size instead -- use
                    it when the volume holds something you want to keep.

NAMESPACE
  The trailing argument, else $LLMDBENCH_TEST_NAMESPACE, else llm-d-$USER.
  One run per namespace -- see --allow-busy-ns.
EOF
}

# -----------------------------------------------------------------------
# Scenarios skipped unless asked for
#
# Everything here needs hardware or an environment a general cluster does not
# have; the reason is printed when one is skipped. Rendering them is harmless,
# so `--plan --no-skip` is the way to keep them covered.
# -----------------------------------------------------------------------
read -r -d '' SKIP_TABLE <<'EOF'
guides/wide-ep|needs RDMA/IB nodes (NCCL + NIXL over InfiniBand)
examples/spyre|needs IBM Spyre accelerators
examples/spyre-s390x|needs IBM Spyre accelerators on s390x
examples/intel-xpu|needs Intel XPU (Gaudi/Arc) nodes
experimental/kimi-k3-h100|experimental: H100 nodes and a very large model
guides/nok8s|deploys containers over ssh, not to Kubernetes
cicd/cks|CI scenario: needs that cluster's infrastructure (run by GitHub Actions)
cicd/gke|CI scenario: needs that cluster's infrastructure (run by GitHub Actions)
cicd/kind|CI scenario: needs that cluster's infrastructure (run by GitHub Actions)
cicd/ocp|CI scenario: needs that cluster's infrastructure (run by GitHub Actions)
cicd/ocp-keda|CI scenario: needs that cluster's infrastructure (run by GitHub Actions)
cicd/ocp-keda-fma-hotstart|CI scenario: needs that cluster's infrastructure (run by GitHub Actions)
cicd/ocp-keda-fma-warmstart|CI scenario: needs that cluster's infrastructure (run by GitHub Actions)
EOF

# The default set: broad enough to cover every deploy path and every engine a
# scenario states, small enough to run on one cluster in one sitting.
DEFAULT_SPECS="examples/cpu
examples/engines
examples/gpu
examples/sim
guides/optimized-baseline
guides/pd-disaggregation
guides/precise-prefix-cache-routing
guides/predicted-latency-routing
guides/tiered-prefix-cache"

# -----------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------
MODE="standup"
SELECT_ALL=false
NS=""
WORKSPACE=""
KUSTOMIZE_BACKEND=""
NO_SKIP=false
ALLOW_BUSY_NS=false
KEEP_PVCS=false
ENGINES=""; METHODS=""; SEL_GROUPS=""; ACCEL=""; SPECS=""; SKIPS=""; EXTRA_SETS=""

append() { # append $2 to the newline-list named $1
  eval "local cur=\$$1"
  if [ -z "$cur" ]; then eval "$1=\$2"; else eval "$1=\"\$cur
\$2\""; fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --plan)     MODE="plan" ;;
    --standup)  MODE="standup" ;;
    --list)     MODE="list" ;;
    --all)      SELECT_ALL=true ;;
    --no-skip)  NO_SKIP=true ;;
    --allow-busy-ns) ALLOW_BUSY_NS=true ;;
    --keep-pvcs) KEEP_PVCS=true ;;
    --engine)   append ENGINES "$2"; shift ;;
    --method)   append METHODS "$2"; shift ;;
    --group)    append SEL_GROUPS "$2"; shift ;;
    --spec)     append SPECS "$2"; shift ;;
    --skip)     append SKIPS "$2"; shift ;;
    --set)      append EXTRA_SETS "$2"; shift ;;
    --accel)    ACCEL="$2"; shift ;;
    --workspace|--ws) WORKSPACE="$2"; shift ;;
    --kustomize-backend) KUSTOMIZE_BACKEND="$2"; append METHODS "kustomize"; shift ;;
    --engine=*|--method=*|--group=*|--spec=*|--skip=*|--set=*|--accel=*|--workspace=*|--kustomize-backend=*)
      key="${1%%=*}"; val="${1#*=}"; set -- "$key" "$val" "${@:2}"; continue ;;
    --help|-h)  usage; exit 0 ;;
    -*)         echo "Unknown option: $1 (use --help)" >&2; exit 1 ;;
    *)          NS="$1" ;;
  esac
  shift
done

NS="${NS:-${LLMDBENCH_TEST_NAMESPACE:-llm-d-$(id -un)}}"
LOG_DIR="${WORKSPACE:-/tmp/llmdbench-test-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR" || exit 1

# `--engine X --method kustomize` means "deploy the upstream guide with X",
# which is what acceleratorBackend selects. A kustomize deploy never reads the
# scenario's own engine.command, so no commented block is switched in either.
if [ -z "$KUSTOMIZE_BACKEND" ] && [ -n "$ENGINES" ] && \
   printf '%s\n' "$METHODS" | grep -qx 'kustomize'; then
  KUSTOMIZE_BACKEND="gpu/$(printf '%s\n' "$ENGINES" | head -1)"
fi

# -----------------------------------------------------------------------
# Selection -- ask the inventory, then apply the skip list
# -----------------------------------------------------------------------
inv_args=""
for e in $(printf '%s\n' "$ENGINES"); do [ -n "$e" ] && inv_args="$inv_args --engine $e"; done
for g in $(printf '%s\n' "$SEL_GROUPS");  do [ -n "$g" ] && inv_args="$inv_args --group $g"; done
[ -n "$ACCEL" ] && inv_args="$inv_args --accel $ACCEL"
# --engine with kustomize selects by capability, not by the scenario's declared
# engine: the backend override is what changes which engine runs.
if [ -n "$KUSTOMIZE_BACKEND" ]; then
  inv_args="$(printf '%s' "$inv_args" | sed 's/--engine [^ ]*//g') --method kustomize"
else
  for m in $(printf '%s\n' "$METHODS"); do [ -n "$m" ] && inv_args="$inv_args --method $m"; done
fi

inventory() { "$PYTHON" "$INVENTORY" --format tsv "$@"; }

ROWS="$(inventory $inv_args)" || {
  echo "Could not read the scenario inventory ($PYTHON $INVENTORY)" >&2; exit 1; }

# `--engine NAME` also selects the scenarios that only offer NAME commented out
# (the inventory's ALT column). Two queries rather than one because the
# inventory ANDs its filters: `--engine X --alternative X` asks for a scenario
# that both launches X and offers it, which is nothing.
if [ -n "$ENGINES" ] && [ -z "$KUSTOMIZE_BACKEND" ]; then
  alt_args="$(printf '%s' "$inv_args" | sed 's/--engine /--alternative /g')"
  ALT_ROWS="$(inventory $alt_args)" || {
    echo "Could not read the scenario inventory ($PYTHON $INVENTORY)" >&2; exit 1; }
  ROWS="$(printf '%s\n%s\n' "$ROWS" "$ALT_ROWS" | grep -v '^$' | awk '!seen[$0]++')"
fi

skip_reason() { printf '%s\n' "$SKIP_TABLE" | grep "^$1|" | head -1 | cut -d'|' -f2-; }
listed()      { printf '%s\n' "$2" | grep -qx "$1"; }
row_for()     { printf '%s\n' "$ROWS" | grep "^$1	" | head -1; }

# The name `--spec` takes. It is a file under config/specification/ that says
# which scenario it reads, so it is not always spelled like the scenario's own
# path -- the inventory reports the real one, and `-` means no specification
# reads this scenario (nothing can run it until one does).
spec_name() { row_for "$1" | cut -f8; }

# The engine `--engine` asked for that this scenario does not launch but does
# offer, as a commented-out `# @engine` group -- i.e. the one the CLI has to
# switch in. Empty when the scenario already launches what was asked for, when
# nothing was asked for, and under kustomize (where acceleratorBackend picks).
switch_for() {
  local row declared alts want
  [ -n "$KUSTOMIZE_BACKEND" ] && return 0
  row="$(row_for "$1")"
  declared="$(printf '%s' "$row" | cut -f5 | tr ',' '\n')"
  alts="$(printf '%s' "$row" | cut -f9 | tr ',' '\n')"
  for want in $(printf '%s\n' "$ENGINES" | awk '!seen[$0]++'); do
    [ -n "$want" ] || continue
    printf '%s\n' "$declared" | grep -qix "$want" && continue
    printf '%s\n' "$alts" | grep -qix "$want" && { printf '%s' "$want"; return 0; }
  done
  return 0
}

# One run per scenario file: the CLI stands up every stack in it.
CANDIDATES="$(printf '%s\n' "$ROWS" | cut -f1 | awk '!seen[$0]++')"

SELECTED=""; SKIPPED=""
for spec in $CANDIDATES; do
  [ -n "$spec" ] || continue
  if [ -n "$SPECS" ]; then
    listed "$spec" "$SPECS" || continue
  elif ! $SELECT_ALL; then
    listed "$spec" "$DEFAULT_SPECS" || continue
  fi
  if listed "$spec" "$SKIPS"; then
    SKIPPED="${SKIPPED}${spec}|asked to skip
"; continue
  fi
  if [ "$(spec_name "$spec")" = "-" ]; then
    SKIPPED="${SKIPPED}${spec}|CANNOT RUN -- no specification reads it: nothing under config/specification/ points at config/scenarios/${spec}.yaml
"; continue
  fi
  reason="$(skip_reason "$spec")"
  if [ -n "$reason" ] && ! $NO_SKIP && ! listed "$spec" "$SPECS"; then
    SKIPPED="${SKIPPED}${spec}|${reason}
"; continue
  fi
  append SELECTED "$spec"
done

# Methods for one scenario: what was asked for (when the scenario can take it),
# else the ones the scenario enables itself.
methods_for() {
  local spec="$1" row declared forcible out=""
  row="$(row_for "$spec")"
  declared="$(printf '%s' "$row" | cut -f3 | tr ',' ' ')"
  forcible="$(printf '%s' "$row" | cut -f4 | tr ',' ' ')"
  if [ -z "$METHODS" ]; then
    printf '%s\n' "$declared" | tr ' ' '\n' | grep -v '^-\?$'
    return
  fi
  for want in $(printf '%s\n' "$METHODS" | awk '!seen[$0]++'); do
    for have in $declared $forcible; do
      [ "$want" = "$have" ] && out="$out $want" && break
    done
  done
  printf '%s\n' "$out" | tr ' ' '\n' | grep -v '^$'
}

bare_engine() { # strip the inventory's `(default)`/`(kust)` marker and ` off`
  printf '%s' "$1" | sed -e 's/([^)]*)//g' -e 's/ *off$//' -e 's/^ *//;s/ *$//'
}

engine_for() { # what engine this (scenario, method) combination exercises
  local spec="$1" method="$2" roles switched
  # The switched command is the one that will run, whatever the file declares.
  switched="$(switch_for "$spec")"
  [ -n "$switched" ] && { printf '%s' "$switched"; return; }
  if [ "$method" = "kustomize" ]; then
    [ -n "$KUSTOMIZE_BACKEND" ] && { printf '%s' "${KUSTOMIZE_BACKEND##*/}"; return; }
  fi
  roles="$(row_for "$spec" | cut -f7)"
  case "$method" in
    standalone) bare_engine "$(printf '%s' "$roles" | tr ',' '\n' | grep '^standalone=' | head -1 | cut -d= -f2)" ;;
    nok8s)      bare_engine "$(printf '%s' "$roles" | tr ',' '\n' | grep '^nok8s=' | head -1 | cut -d= -f2)" ;;
    *)          bare_engine "$(row_for "$spec" | cut -f5)" ;;
  esac
}

# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------
RESULTS=""
record() { RESULTS="${RESULTS}$1|$2|$3
"; }

cli() { # cli <spec-arg> <phase> <method> <logfile> [extra args...]
  local spec="$1" phase="$2" method="$3" log="$4"; shift 4
  local args="--spec $spec"
  [ "$phase" = "plan" ] && args="$args --dry-run"
  set -- "$phase" -p "$NS" -t "$method" --ws "${LOG_DIR}/ws" "$@"
  [ -n "$KUSTOMIZE_BACKEND" ] && [ "$method" = "kustomize" ] && \
    set -- "$@" --set "kustomize.acceleratorBackend=${KUSTOMIZE_BACKEND}"
  for kv in $(printf '%s\n' "$EXTRA_SETS"); do
    [ -n "$kv" ] && set -- "$@" --set "$kv"
  done
  llmdbenchmark $args "$@" 2>&1 | tee "$log"
  return ${PIPESTATUS[0]}
}

# -----------------------------------------------------------------------
# The model PVC the previous scenario left behind
#
# A model PVC deliberately outlives teardown: the next standup finds the weights
# already staged instead of downloading them again, which is most of a standup's
# wall clock. That is a win while the scenarios want the same size, and a hard
# stop when one wants more -- `examples/engines` takes the 300Gi default,
# `guides/optimized-baseline` asks for 1Ti, and run in that order the second
# standup ends with
#     PVC 'model-pvc' exists with size 300Gi but 1Ti is required
# A bound PVC cannot be grown from here (a StorageClass need not allow expansion,
# and another model's weights are sitting in it), so a sequence of scenarios only
# runs unattended if the small one is deleted first.
#
# So this DELETES an undersized model PVC before the standup that would trip over
# it, and says so each time. It costs that scenario a fresh download. --keep-pvcs
# turns it off: the standup then fails on the size, which is what you want when
# the volume holds something you would rather not lose. Only *model* volumes are
# considered -- the workload PVC holds results and its size does not vary.
# -----------------------------------------------------------------------

# `name<TAB>size` for every model PVC a rendered plan asks for. Largest size per
# name: the stacks of one scenario may state different sizes for a shared volume,
# and the largest is the one all of them need.
plan_model_pvcs() { # plan_model_pvcs <plan-dir>
  "$PYTHON" - "$1" <<'PY'
import pathlib
import sys

import yaml

from llmdbenchmark.executor.step import Step

wanted: dict[str, tuple[float, str]] = {}
for cfg in sorted(pathlib.Path(sys.argv[1]).glob("*/config.yaml")):
    storage = (yaml.safe_load(cfg.read_text()) or {}).get("storage") or {}
    for key in ("modelPvc", "extraPvc"):
        pvc = storage.get(key) or {}
        name, size = str(pvc.get("name") or ""), str(pvc.get("size") or "")
        gi = Step._parse_size_gi(size) if size else None
        if not name or gi is None:
            continue
        if name not in wanted or gi > wanted[name][0]:
            wanted[name] = (gi, size)
for name, (_, size) in sorted(wanted.items()):
    print(f"{name}\t{size}")
PY
}

# Is what is there smaller than what the plan asks for? The comparison is the
# standup's own, imported rather than reimplemented, so this cannot disagree with
# the check it exists to get ahead of (llmdbenchmark/executor/step.py).
pvc_too_small() { # pvc_too_small <existing-size> <required-size>
  "$PYTHON" - "$1" "$2" <<'PY'
import sys

from llmdbenchmark.executor.step import Step

have, need = (Step._parse_size_gi(arg) for arg in sys.argv[1:3])
sys.exit(0 if have is not None and need is not None and have < need else 1)
PY
}

prepare_model_pvcs() { # prepare_model_pvcs <spec-arg> <method> <safe> [cli args...]
  local spec_arg="$1" method="$2" safe="$3"; shift 3
  [ "$KEEP_PVCS" = true ] && return 0
  [ -n "$KUBECTL" ] || return 0

  # Nothing staged here means nothing can be in the way. One call, and it is the
  # answer on a fresh namespace, so it comes before rendering anything.
  local existing
  existing="$("$KUBECTL" get pvc -n "$NS" \
                -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)"
  [ -n "$existing" ] || return 0

  # What this scenario asks for, read from its own rendered plan: the sizes come
  # from defaults.yaml as often as from the scenario file, so rendering is the
  # only honest way to know them.
  local plan_log="${LOG_DIR}/pvcplan-${safe}.log"
  if ! cli "$spec_arg" plan "$method" "$plan_log" "$@" >/dev/null 2>&1; then
    echo "  Note: could not render ${spec_arg} to compare PVC sizes"
    echo "        (${plan_log}); leaving the existing PVCs alone."
    return 0
  fi

  local pvcs name need have
  pvcs="$(plan_model_pvcs "${LOG_DIR}/ws/latest/plan" 2>/dev/null)"
  [ -n "$pvcs" ] || return 0

  while IFS="$(printf '\t')" read -r name need; do
    [ -n "$name" ] || continue
    printf '%s\n' "$existing" | grep -qx "$name" || continue
    have="$("$KUBECTL" get pvc "$name" -n "$NS" \
              -o 'jsonpath={.spec.resources.requests.storage}' 2>/dev/null)"
    [ -n "$have" ] || continue
    pvc_too_small "$have" "$need" || continue
    echo ""
    echo "  pvc/${name} in ns/${NS} is ${have}, and ${spec_arg} needs ${need}."
    echo "  A model PVC survives teardown on purpose, so this one belongs to an"
    echo "  earlier scenario. It cannot be grown, so DELETING it -- the weights"
    echo "  in it are downloaded again, which is the slow part of this standup."
    echo "  (--keep-pvcs leaves it alone; the standup then fails on the size.)"
    "$KUBECTL" delete pvc "$name" -n "$NS" --wait=true 2>&1 | sed 's/^/    /'
    echo ""
  done <<EOF
$pvcs
EOF
}

run_one() {
  local spec="$1" method="$2" engine="$3" switched="$4"
  local label="${spec} [${method}/${engine:-?}]"
  local safe; safe="$(printf '%s' "${spec}-${method}-${engine}" | tr '/' '-')"
  local spec_arg; spec_arg="$(spec_name "$spec")"
  # The CLI writes the switched scenario under --ws and renders that, so the
  # flag has to be on every phase: teardown reads the same rendered plan.
  local sw=(); [ -n "$switched" ] && sw=(--engine "$switched")

  echo "=========================================="
  echo "Testing: ${label}"
  echo "=========================================="

  if [ "$MODE" = "plan" ]; then
    cli "$spec_arg" plan "$method" "${LOG_DIR}/plan-${safe}.log" "${sw[@]}"
    if [ $? -eq 0 ]; then echo "PLAN PASSED: ${label}"; record PASS "$label" plan
    else echo "PLAN FAILED: ${label}"; record FAIL "$label" plan; fi
    echo ""
    return
  fi

  # An undersized model PVC left by the scenario before this one stops the
  # standup -- after the download it has already invalidated. Clear it first.
  prepare_model_pvcs "$spec_arg" "$method" "$safe" "${sw[@]}"

  cli "$spec_arg" standup "$method" "${LOG_DIR}/standup-${safe}.log" "${sw[@]}"
  if [ $? -eq 0 ]; then echo "STANDUP PASSED: ${label}"; record PASS "$label" standup
  else echo "STANDUP FAILED: ${label}"; record FAIL "$label" standup; fi

  # Teardown always runs, so a failed standup does not strand the namespace.
  cli "$spec_arg" teardown "$method" "${LOG_DIR}/teardown-${safe}.log" "${sw[@]}"
  if [ $? -eq 0 ]; then record PASS "$label" teardown
  else echo "TEARDOWN FAILED: ${label}"; record FAIL "$label" teardown; fi
  echo ""
}

# -----------------------------------------------------------------------
# Is the namespace free?
#
# Deployment names are derived from the stack name and the model shortName, so a
# second run of the same scenario in the same namespace does not deploy beside
# the first -- it replaces its pods. Both runs then validate against whichever
# rollout landed last, and the loser fails on checks its own plan rendered
# correctly (a missing `LD_LIBRARY_PATH`, an engine command that "does not start
# in the container args at all"). The plan on disk is right; the pod is somebody
# else's. Refuse up front instead, and say who is already there.
#
# `configmap/llm-d-benchmark-standup-parameters` is written at the end of a
# standup and deleted by teardown, so it carries who and when. Engine pods are
# the more reliable signal -- a killed run leaves them behind with no configmap
# -- so either one is enough to stop.
# -----------------------------------------------------------------------
KUBECTL="$(command -v kubectl || command -v oc)"
HELM="$(command -v helm)"

# What "already in use" looks like, in the order standup creates it. Engine pods
# are the *last* of these to appear, so a check that only looks for them sees an
# idle namespace for the several minutes a model download takes -- which is
# exactly when the collision below was let through.
preflight_ns() {
  [ -n "$KUBECTL" ] || {
    echo "Note: no kubectl/oc on PATH -- skipping the namespace pre-flight check."
    return 0
  }

  local releases downloads pods owner found=""
  # [07/08] Helm releases -- the earliest and the hardest evidence. A release
  # owns its objects, and some of their names are fixed by the upstream chart
  # rather than derived from the stack: `epponly` renders a ConfigMap named
  # literally `envoy` (router.proxy.presets.envoy.configMap.name), so the second
  # release in a namespace fails its install outright with "invalid ownership
  # metadata ... must equal <release>". A *failed* release still owns them, so
  # these count even when nothing is running.
  [ -n "$HELM" ] && releases="$("$HELM" list -n "$NS" -q 2>/dev/null)"
  # [04] Weights staging: minutes long, and the only thing in the namespace for
  # most of it.
  downloads="$("$KUBECTL" get jobs,pods -n "$NS" \
                 -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
               | grep '^download-model' || true)"
  # [08] The engine pods themselves.
  pods="$("$KUBECTL" get pods -n "$NS" -l llm-d.ai/role \
            -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)"

  found="$(printf '%s\n%s\n%s\n' "$releases" "$downloads" "$pods" | grep -v '^$' || true)"
  # A namespace that does not exist yet, or that we cannot read, is not a
  # collision -- standup reports either far better than a pre-flight can.
  [ -n "$found" ] || return 0

  owner="$("$KUBECTL" get configmap llm-d-benchmark-standup-parameters -n "$NS" \
             -o 'jsonpath={.data.deployed_by} at {.data.deployed_at} (release {.data.release})' \
             2>/dev/null)"
  echo ""
  echo "  ns/${NS} is already in use:"
  [ -n "$releases" ] && printf '    helm release  %s\n' $releases
  [ -n "$downloads" ] && printf '    downloading   %s\n' $downloads
  [ -n "$pods" ] && printf '    engine pod    %s\n' $pods
  [ -n "$owner" ] && echo "    stood up by ${owner}"
  if [ "$ALLOW_BUSY_NS" = true ]; then
    echo "  --allow-busy-ns given: continuing. Expect a failed router install," >&2
    echo "  replaced pods, or validation failures against a correct plan." >&2
    return 0
  fi
  cat >&2 <<EOM

  Refusing to start. A second deployment here fails in one of two ways, neither
  of which reads as what it is:
    * the router install is rejected -- "ConfigMap envoy ... invalid ownership
      metadata", because that name belongs to the release already here;
    * or it installs, renders the same Deployment names, replaces those pods,
      and validation then inspects the winner's pods against the loser's plan.

  Either wait for that run to finish (or tear it down), or use another
  namespace:  ./util/test-scenarios.sh <other-namespace>
  --allow-busy-ns overrides this.
EOM
  exit 1
}

# -----------------------------------------------------------------------
# What is about to happen
# -----------------------------------------------------------------------
PLANNED=""
SWITCHED=""
for spec in $(printf '%s\n' "$SELECTED"); do
  [ -n "$spec" ] || continue
  switched="$(switch_for "$spec")"
  [ -n "$switched" ] && SWITCHED="${SWITCHED}${spec} -> ${switched}
"
  for method in $(methods_for "$spec"); do
    PLANNED="${PLANNED}${spec}	${method}	$(engine_for "$spec" "$method")	${switched}
"
  done
done

echo "=========================================="
case "$MODE" in
  plan) echo "Test Suite: render only (no cluster)"
        echo "Note:       chart versions stay \"auto\" without a cluster, so the"
        echo "            post-render \`helmfile template\` check logs an error"
        echo "            (\"improper constraint: auto\") on every scenario. That"
        echo "            is expected here -- PASS/FAIL below is the render." ;;
  list) echo "Test Suite: selection only" ;;
  *)    echo "Test Suite: Standup/Teardown Validation" ;;
esac
[ "$MODE" = "list" ] || echo "Namespace:  ${NS}"
echo "Workspace:  ${LOG_DIR}"
[ -n "$KUSTOMIZE_BACKEND" ] && echo "Kustomize:  acceleratorBackend=${KUSTOMIZE_BACKEND}"
echo "=========================================="
printf 'SCENARIO\tMETHOD\tENGINE\n'
printf '%s' "$PLANNED" | grep -v '^$' | sort | cut -f1-3
count="$(printf '%s' "$PLANNED" | grep -cv '^$')"
echo "-- ${count} run(s)"
if [ -n "$SWITCHED" ]; then
  echo ""
  echo "Switched in from a commented-out \`# @engine\` block (workspace only):"
  printf '%s' "$SWITCHED" | grep -v '^$' | sort -u | sed 's/^/  /'
fi
if [ -n "$SKIPPED" ]; then
  echo ""
  echo "Skipped (--no-skip to include, or name one with --spec):"
  printf '%s' "$SKIPPED" | grep -v '^$' | while IFS='|' read -r s r; do
    printf '  %-42s %s\n' "$s" "$r"
  done
fi
echo ""

# A selection that matches nothing is a failure in every mode: asking for a
# scenario or engine that is not there should not look like a clean run.
if [ "$count" -eq 0 ]; then
  echo "Nothing selected. Try --list --all to see what exists." >&2
  exit 1
fi
if [ "$MODE" = "list" ]; then exit 0; fi

# After the selection is printed (so a refusal still shows what would have run)
# and only for modes that deploy: --plan touches no cluster.
[ "$MODE" = "plan" ] || preflight_ns

printf '%s' "$PLANNED" | grep -v '^$' | sort > "${LOG_DIR}/planned.tsv"
while IFS='	' read -r spec method engine switched; do
  [ -n "$spec" ] || continue
  run_one "$spec" "$method" "$engine" "$switched"
done < "${LOG_DIR}/planned.tsv"

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
pass_count=0; fail_count=0
printf '%s' "$RESULTS" | grep -v '^$' | while IFS='|' read -r status label phase; do
  if [ "$status" = "PASS" ]; then printf '  PASS  %s (%s)\n' "$label" "$phase"
  else printf '  FAIL  %s (%s)\n' "$label" "$phase"; fi
done
pass_count="$(printf '%s' "$RESULTS" | grep -c '^PASS|')"
fail_count="$(printf '%s' "$RESULTS" | grep -c '^FAIL|')"

echo ""
echo "Total: $((pass_count + fail_count)) checks, ${pass_count} passed, ${fail_count} failed"
echo "Logs:  ${LOG_DIR}"
echo "=========================================="

[ "$fail_count" -eq 0 ]
