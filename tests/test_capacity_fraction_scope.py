"""One memory fraction, three denominators, three different verdicts.

Every engine takes a fraction of GPU memory and every engine means something
else by it. The pre-deploy capacity check read all of them as vLLM's, and on the
numbers below -- Qwen3-32B on one 80 GiB device at 0.88, taken verbatim from a
live `guides/optimized-baseline [modelservice/sglang]` standup -- that is not a
rounding error:

    device scope (vLLM)      70.40 claimed - 61.02 weights - 5.75 overhead
                             =  3.63 GB KV  <  3.91 GB for one request
                             -> "DEPLOYMENT WILL FAIL: cannot serve any requests"
    weightsAndKv (SGLang)    70.40 static   - 61.02 weights
                             =  9.38 GB KV  -> two concurrent requests, and the
                                overhead fits in the 9.60 GB left outside
    freeAfterLoad (TRT-LLM)  (80 - 61.02 - 5.75) x 0.88
                             = 11.64 GB KV  -> understated 3x by vLLM's reading

The SGLang deployment that verdict condemned was already serving. So these tests
pin each engine's arithmetic to its own flag's definition, pin the vLLM path to
`planner.allocatable_kv_cache_memory` so that verdict stays exactly what it was,
and cover the failure only SGLang can have: a fraction so high that the KV pool
is fine and the forward pass has nowhere to run.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from llmdbenchmark.engine.spec import (
    ENGINE_SPECS,
    MEMORY_FRACTION_DEVICE,
    MEMORY_FRACTION_FREE_AFTER_LOAD,
    MEMORY_FRACTION_SCOPES,
    MEMORY_FRACTION_WEIGHTS_AND_KV,
    GENERIC,
    get_engine_spec,
)
from llmdbenchmark.utilities import capacity_validator as cv

#: The live standup's numbers, per GPU, in GiB.
WEIGHTS = 61.02
ACTIVATION = 5.60
NON_TORCH = 0.15
CUDA_GRAPH = 0.0
INTERMEDIATES = ACTIVATION + CUDA_GRAPH + NON_TORCH  # 5.75
PER_REQUEST_KV = 3.91  # Qwen3-32B at max_model_len=16000
GPU_MEMORY = 80
MODEL = "Qwen/Qwen3-32B"


class _Config(SimpleNamespace):
    """Stands in for an `AutoConfig`; only its identity is used once stubbed."""


@pytest.fixture
def planner(monkeypatch):
    """Replace the planner's model and hardware estimates with fixed numbers.

    Every one of them reaches out to HuggingFace or to an empirical profile
    table, and none of them is what these tests are about: the subtraction is.
    Patched on `capacity_validator`, which imported the names directly.
    """
    monkeypatch.setattr(cv, "model_memory_req", lambda *a, **k: WEIGHTS)
    monkeypatch.setattr(
        cv, "estimate_vllm_activation_memory", lambda *a, **k: ACTIVATION
    )
    monkeypatch.setattr(cv, "estimate_vllm_non_torch_memory", lambda *a, **k: NON_TORCH)
    monkeypatch.setattr(
        cv, "estimate_vllm_cuda_graph_memory", lambda *a, **k: CUDA_GRAPH
    )
    monkeypatch.setattr(cv, "model_total_params", lambda *a, **k: 32_762_123_264)
    monkeypatch.setattr(cv, "find_possible_tp", lambda *a, **k: [1, 2, 4, 8])
    monkeypatch.setattr(cv, "max_context_len", lambda *a, **k: 40960)
    monkeypatch.setattr(cv, "get_text_config", lambda config: config)
    monkeypatch.setattr(cv, "get_model_config_from_hf", lambda *a, **k: _Config())
    monkeypatch.setattr(
        cv,
        "KVCacheDetail",
        lambda *a, **k: SimpleNamespace(per_request_kv_cache_gb=PER_REQUEST_KV),
    )


def _params(engine: str, fraction: float = 0.88, **over) -> cv.ValidationParams:
    kwargs = dict(
        models=[MODEL],
        hf_token=None,
        replicas=1,
        gpu_memory=GPU_MEMORY,
        tp=1,
        pp=1,
        dp=1,
        accelerator_nr=1,
        gpu_memory_util=fraction,
        max_model_len=16000,
        engine=engine,
        ignore_failures=True,
        label="decode",
    )
    kwargs.update(over)
    return cv.ValidationParams(**kwargs)


def _run(params) -> list[str]:
    import logging

    return cv.validate_vllm_params(
        params, cv._ensure_logger(logging.getLogger(__name__))
    )


def _budget(engine: str, fraction: float = 0.88, **over):
    params = _params(engine, fraction, **over)
    return cv._memory_budget(params, MODEL, _Config(), get_engine_spec(engine))


# --------------------------------------------------------------------------
# the arithmetic, one scope at a time
# --------------------------------------------------------------------------


def test_vllm_takes_activations_out_of_the_fraction(planner):
    """`--gpu-memory-utilization` is the whole device budget."""
    b = _budget("vllm")
    assert b.scope == MEMORY_FRACTION_DEVICE
    assert b.claimed == pytest.approx(70.4)
    assert b.kv == pytest.approx(70.4 - WEIGHTS - INTERMEDIATES)  # 3.63
    assert b.kv < PER_REQUEST_KV


def test_sglang_takes_activations_out_of_what_the_fraction_left(planner):
    """`--mem-fraction-static` is weights + KV pool; the rest is on top."""
    b = _budget("sglang")
    assert b.scope == MEMORY_FRACTION_WEIGHTS_AND_KV
    assert b.kv == pytest.approx(70.4 - WEIGHTS)  # 9.38
    assert b.outside == pytest.approx(80 - 70.4)  # 9.60 for the intermediates
    assert b.intermediates_fit_outside
    assert b.kv > PER_REQUEST_KV


def test_trtllm_takes_its_fraction_of_what_is_left_after_loading(planner):
    """`--kv_cache_free_gpu_memory_fraction` is a fraction of free memory."""
    b = _budget("trtllm")
    assert b.scope == MEMORY_FRACTION_FREE_AFTER_LOAD
    assert b.kv == pytest.approx((80 - WEIGHTS - INTERMEDIATES) * 0.88)  # 11.64
    # The reading that was applied to it before: 3x smaller, and a failure.
    assert b.kv > _budget("vllm").kv * 3


def test_an_alias_gets_its_engines_scope(planner):
    """`tensorrt-llm` is TRT-LLM, so it is TRT-LLM's arithmetic."""
    assert _budget("tensorrt-llm").kv == pytest.approx(_budget("trtllm").kv)


def test_an_unknown_engine_keeps_the_reading_this_check_always_used(planner):
    """Generic is the fallback, and nothing is known about its fraction.

    vLLM's reading is the conservative one of the three (it subtracts the most),
    and it is what every engine got before the scope was recorded.
    """
    assert _budget("someengine").kv == pytest.approx(_budget("vllm").kv)
    assert _budget("").kv == pytest.approx(_budget("vllm").kv)


def test_the_widths_scale_the_whole_group(planner):
    """TP x PP x DP devices, weights per DP rank -- the planner's scaling."""
    b = _budget("vllm", tp=2, dp=2)
    assert b.total == pytest.approx(GPU_MEMORY * 4)
    assert b.weights == pytest.approx(WEIGHTS * 2)
    assert b.intermediates == pytest.approx(ACTIVATION * 2 + NON_TORCH * 4)


def test_the_device_scope_reproduces_the_planner(monkeypatch, planner):
    """vLLM's number must stay the planner's number, or this refactor moved it.

    The planner's own module-level names are patched with the same estimates, so
    the two are compared on identical inputs; the planner clamps at zero and this
    does not, hence the `max`.
    """
    from planner import capacity_planner as pl

    monkeypatch.setattr(pl, "model_memory_req", lambda *a, **k: WEIGHTS)
    monkeypatch.setattr(
        pl, "estimate_vllm_activation_memory", lambda *a, **k: ACTIVATION
    )
    monkeypatch.setattr(pl, "estimate_vllm_non_torch_memory", lambda *a, **k: NON_TORCH)
    monkeypatch.setattr(
        pl, "estimate_vllm_cuda_graph_memory", lambda *a, **k: CUDA_GRAPH
    )

    for tp, pp, dp, frac in ((1, 1, 1, 0.88), (2, 1, 1, 0.9), (2, 1, 2, 0.75)):
        mine = _budget("vllm", frac, tp=tp, pp=pp, dp=dp)
        theirs = pl.allocatable_kv_cache_memory(
            MODEL,
            _Config(),
            GPU_MEMORY,
            frac,
            tp=tp,
            pp=pp,
            dp=dp,
            max_model_len=16000,
            batch_size=1,
        )
        assert max(0.0, mine.kv) == pytest.approx(theirs)


# --------------------------------------------------------------------------
# the verdicts
# --------------------------------------------------------------------------


def test_the_false_alarm_that_started_this_is_gone(planner):
    """The SGLang stack this check condemned was already serving requests."""
    messages = _run(_params("sglang"))
    assert not [m for m in messages if "WILL FAIL" in m], messages
    assert any("9.38 GB for KV" in m for m in messages), messages
    # floor(9.38 / 3.91)
    assert any(
        "Max concurrent requests (worst case, each at max_model_len): 2" in m
        for m in messages
    )


def test_vllms_verdict_on_the_same_numbers_is_unchanged(planner):
    """Not a loosened check: on vLLM's own reading these numbers still fail."""
    messages = _run(_params("vllm"))
    assert any("cannot serve any requests" in m for m in messages), messages
    assert any("Available KV cache: 3.63 GB" in m for m in messages), messages


def test_trtllm_passes_where_the_vllm_reading_failed_it(planner):
    messages = _run(_params("trtllm"))
    assert not [m for m in messages if "WILL FAIL" in m], messages
    assert any("11.64 GB for KV" in m for m in messages), messages


def test_too_high_a_static_fraction_fails_on_the_activations_not_the_pool(planner):
    """SGLang's own failure mode: the KV pool is ample, the forward pass OOMs.

    0.95 on an 80 GiB device leaves 4.00 GB outside the static allocation, and
    activations plus overhead need 5.75 GB. The KV pool is 14.98 GB -- nearly
    four requests' worth -- so no pool-based check can see this.
    """
    messages = _run(_params("sglang", 0.95))
    assert any("no room left for activations" in m for m in messages), messages
    assert any("need 5.75 GB on top of it" in m for m in messages), messages
    # The fix is to *lower* the fraction, which is the opposite of what the
    # vLLM-shaped advice says: (80 - 5.75) / 80.
    assert any("Reduce --mem-fraction-static to at most 0.93" in m for m in messages), (
        messages
    )
    assert not [m for m in messages if "Increase the memory fraction" in m], messages


def test_a_model_that_does_not_fit_says_so_rather_than_reporting_an_empty_pool(planner):
    """The planner clamps a negative pool to 0, which reads as "loads but ...".

    A pool of exactly zero is not a servable deployment either way, so the two
    branches differ only in which message a user gets -- and "insufficient memory
    to load" is the true one.
    """
    messages = _run(_params("vllm", 0.7))  # 56 - 61.02 - 5.75 < 0
    assert any("Insufficient GPU memory to load model" in m for m in messages), messages
    assert any("10.77 GB MORE" in m for m in messages), messages


def test_the_suggestions_name_the_flag_the_user_actually_wrote(planner):
    """`gpu_memory_utilization` is not a flag SGLang or TRT-LLM has."""
    messages = _run(_params("sglang", 0.5))  # pool too small: 40 - 61.02 < 0
    assert any("--mem-fraction-static: 0.5" in m for m in messages), messages
    assert any("engine: sglang" in m for m in messages), messages
    assert not [m for m in messages if "gpu_memory_utilization" in m], messages


def test_raising_a_free_memory_fraction_is_advised_toward_one(planner):
    """TRT-LLM's fraction is of free memory: 1.0 is its ceiling, not an OOM."""
    messages = _run(_params("trtllm", 0.1))  # 13.23 x 0.1 = 1.32 < 3.91
    assert any("cannot serve any requests" in m for m in messages), messages
    assert any("toward 1.0" in m for m in messages), messages


def test_an_unstated_fraction_skips_the_estimate_and_names_the_right_flag(planner):
    messages = _run(_params("sglang", 0.0))
    assert any(
        "sglang command does not set a GPU memory fraction (--mem-fraction-static)" in m
        for m in messages
    ), messages
    assert not [m for m in messages if "WILL FAIL" in m], messages


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", (*ENGINE_SPECS, GENERIC), ids=lambda s: s.name)
def test_every_engine_states_a_scope_the_check_understands(spec):
    """A new engine whose scope is a typo would silently get vLLM's reading."""
    assert spec.memory_fraction_scope in MEMORY_FRACTION_SCOPES


@pytest.mark.parametrize("spec", ENGINE_SPECS, ids=lambda s: s.name)
def test_an_engine_that_reads_a_fraction_is_covered_by_a_test_above(spec):
    """Each scope in use is pinned by one of the arithmetic tests.

    Kept as an assertion rather than a comment: adding an engine with a fraction
    flag and a fourth scope should fail here until its arithmetic is written.
    """
    if not spec.memory_util_flags:
        return
    assert spec.memory_fraction_scope in {
        MEMORY_FRACTION_DEVICE,
        MEMORY_FRACTION_WEIGHTS_AND_KV,
        MEMORY_FRACTION_FREE_AFTER_LOAD,
    }


def test_the_plan_supplies_the_engine_to_validate_with():
    """`_extract_params` reads the role's detected engine, not a default."""
    plan = {
        "model": {
            "huggingfaceId": MODEL,
            "gpuMemoryUtilization": 0.88,
            "maxModelLen": 16000,
        },
        "accelerator": {"type": "NVIDIA-A100-SXM4-80GB", "memory": "80"},
        "decode": {
            "replicas": 1,
            "acceleratorNr": 1,
            "engine": {"name": "sglang"},
        },
    }
    params = cv._extract_params(plan, "decode", ignore_failures=True)
    assert params is not None
    assert params.engine == "sglang"
    assert get_engine_spec(params.engine).memory_fraction_scope == (
        MEMORY_FRACTION_WEIGHTS_AND_KV
    )


def test_a_role_with_no_engine_block_still_validates():
    """An older rendered plan, or a role the resolver left alone."""
    plan = {
        "model": {"huggingfaceId": MODEL, "maxModelLen": 16000},
        "accelerator": {"memory": "80"},
        "decode": {"replicas": 1, "acceleratorNr": 1},
    }
    params = cv._extract_params(plan, "decode", ignore_failures=True)
    assert params is not None and params.engine == ""


def test_concurrency_matches_the_planners_own_floor_division(planner):
    """The estimate is `floor(pool / per request)`, only of this engine's pool."""
    b = _budget("sglang")
    messages = _run(_params("sglang"))
    expected = math.floor(b.kv / PER_REQUEST_KV)
    assert any(f"each at max_model_len): {expected}" in m for m in messages)
