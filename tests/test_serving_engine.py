"""``serving_engine``: which engine a health check is about to dial.

The smoketest's health check used to log "verifying vLLM is listening" and poll
a hardcoded ``/health`` whatever the stack ran. Both facts are in the plan --
``resolve_engines`` writes ``name`` and ``healthPath`` per role from whichever
launcher the scenario's command names -- and it is one endpoint being dialled,
so the engine behind it is a single well-defined thing worth reading rather than
assuming.

The interesting part is the fallback chain: a nok8s or standalone scenario has
no ``decode`` at all, a disabled role still leaves a stub behind in the rendered
config, and a plan with no per-role engine has to land on something rather than
report the wrong engine's path.
"""

from __future__ import annotations

from llmdbenchmark.engine import serving_engine, serving_port


VLLM = {"name": "vllm", "healthPath": "/health", "command": "vllm serve m --port 8200"}
TRTLLM = {
    "name": "trtllm",
    "healthPath": "/health",
    "metricsPath": "/prometheus/metrics",
    "command": "trtllm-serve serve m --port 8200",
}


def test_decode_wins_on_a_disaggregated_stack():
    """Both roles serve; decode is the one behind the inference endpoint."""
    cfg = serving_engine({"decode": {"engine": VLLM}, "prefill": {"engine": TRTLLM}})
    assert cfg["name"] == "vllm"


def test_a_standalone_only_stack_is_found():
    cfg = serving_engine({"standalone": {"engine": TRTLLM}})
    assert cfg["name"] == "trtllm"
    assert cfg["metricsPath"] == "/prometheus/metrics"


def test_a_nok8s_stack_is_found():
    assert serving_engine({"nok8s": {"engine": VLLM}})["name"] == "vllm"


def test_a_role_left_behind_with_no_engine_is_skipped():
    """A disabled role still renders its section, so an empty `engine` block --
    or a missing one -- must not shadow the role that actually serves."""
    values = {
        "prefill": {"replicas": 0, "engine": {}},
        "decode": {"engine": TRTLLM},
        "standalone": {"enabled": False},
    }
    assert serving_engine(values)["name"] == "trtllm"


def test_the_plan_wide_block_is_the_fallback():
    """`common.engine` renders here and carries the pod shape, not a launcher --
    but a config with no per-role engine at all should still yield its paths."""
    cfg = serving_engine({"engine": {"healthPath": "/v1/health", "shmMemory": "16Gi"}})
    assert cfg["healthPath"] == "/v1/health"


def test_nothing_resolvable_is_an_empty_dict_not_an_error():
    """The caller defaults to "the engine" and /health from this; raising here
    would turn a missing key into a failed standup."""
    assert serving_engine({}) == {}
    assert serving_engine({"decode": {"replicas": 1}}) == {}
    assert serving_engine({"engine": "not-a-mapping"}) == {}


# ---------------------------------------------------------------------------
# serving_port: the port a probe should dial on a pod
# ---------------------------------------------------------------------------
#
# Regression: an sglang standalone run on a real cluster came up healthy on
# 8200 (its command says `--port 8200`) and the smoketest probed the pod on
# 8000, because it read the plan-wide `engine.servicePort` -- the *Service*'s
# port, whose default is 8000. The command is the only thing that knows which
# port the process binds, so the resolved engine block is the only right
# source for reaching a pod directly.

SGLANG_8200 = {
    "name": "sglang",
    "port": 8200,
    "healthPath": "/health",
    "command": "python3 -m sglang.launch_server --model-path m --port 8200",
}


def test_standalone_port_comes_from_the_command_not_the_service():
    values = {
        "standalone": {"engine": SGLANG_8200},
        "engine": {"servicePort": 8000},
    }
    assert serving_port(values) == 8200


def test_decode_port_wins_for_a_modelservice_stack():
    values = {"decode": {"engine": SGLANG_8200}, "engine": {"servicePort": 8000}}
    assert serving_port(values) == 8200


def test_falls_back_to_the_service_port_with_no_resolved_engine():
    """A bare values tree (`--dry-run`, an unresolved plan) still answers."""
    assert serving_port({"engine": {"servicePort": 8001}}) == 8001


def test_falls_back_past_an_unusable_port():
    values = {
        "standalone": {"engine": {"name": "x", "port": "", "command": "x serve m"}},
        "engine": {"servicePort": 8080},
    }
    assert serving_port(values) == 8080


def test_answers_with_nothing_to_go_on():
    assert serving_port({}) == 8000
