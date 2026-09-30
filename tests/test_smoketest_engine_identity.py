"""``assert_engine_identity`` asks the pod, not the plan.

The check it replaces compared ``<role>.engine.name`` -- which ``resolve_engines``
wrote by reading the scenario's own command -- against the name of a container it
had just found in the pod, and printed "decode runs engine 'trtllm'" on the
strength of it. That sentence cannot be false for any pod this harness renders,
so it passed against a pod running SGLang while the TRT-LLM plan beside it was
perfectly correct.

That was not hypothetical: two standups of ``guides/optimized-baseline`` in one
namespace render the same Deployment name, the second rollout replaced the
first's pods, and the losing run's validation failed on ``engine_command`` and a
missing ``LD_LIBRARY_PATH`` while cheerfully reporting the right engine.

So the facts here all come from the pod: the launcher in its args (via the same
``detect_engine`` the resolver used) and the image it runs.
"""

from __future__ import annotations

import pytest

from llmdbenchmark.smoketests.base import BaseSmoketest


PREPROCESS = "python3 /setup/preprocess/set_llmdbench_environment.py && . $HOME/llmdbench_env.sh ;"

#: (engine, image repository, launch line) as the three engines actually render.
ENGINES = [
    (
        "vllm",
        "docker.io/vllm/vllm-openai",
        "vllm serve Qwen/Qwen3-32B --host 0.0.0.0 --port 8200 --max-model-len 16000",
    ),
    (
        "sglang",
        "docker.io/lmsysorg/sglang",
        "python3 -m sglang.launch_server --model-path Qwen/Qwen3-32B --port 8200 "
        "--context-length 16000 --mem-fraction-static 0.88",
    ),
    (
        "trtllm",
        "nvcr.io/nvidia/tensorrt-llm/release",
        "mkdir -p /tmp/trtllm ; trtllm-serve serve Qwen/Qwen3-32B --port 8200 "
        "--backend pytorch --max_seq_len 16000",
    ),
]


def _pod(command: str, image: str, container: str = "modelserver") -> dict:
    """A pod spec shaped like the one the chart renders: shell -c <command>."""
    return {
        "spec": {
            "containers": [
                {
                    "name": container,
                    "image": image,
                    "args": ["-c", f"{PREPROCESS} {command}"],
                },
                {
                    "name": "routing-proxy",
                    "image": "ghcr.io/llm-d/llm-d-routing-sidecar",
                },
            ]
        }
    }


@pytest.mark.parametrize(
    ("engine", "repo", "command"), ENGINES, ids=lambda v: str(v)[:24]
)
def test_a_matching_pod_passes_and_says_what_it_verified(engine, repo, command):
    result = BaseSmoketest.assert_engine_identity(
        _pod(command, f"{repo}:v1.2.3"), engine, expected_repository=repo
    )
    assert result.passed, result.message
    assert "launcher in args" in result.message
    assert repo in result.message
    assert "container 'modelserver'" in result.message


def test_the_collision_that_started_this_fails():
    """An SGLang pod inspected against a TRT-LLM plan -- the real failure."""
    _, sglang_repo, sglang_cmd = ENGINES[1]
    result = BaseSmoketest.assert_engine_identity(
        _pod(sglang_cmd, f"{sglang_repo}:v0.5.19"),
        "trtllm",
        expected_repository="nvcr.io/nvidia/tensorrt-llm/release",
        pod_name="qwen-decode-66448fc968-jdgtp",
    )
    assert not result.passed
    # Both halves named: what is running and what was expected.
    assert "sglang" in result.message and "trtllm" in result.message
    assert "nvcr.io/nvidia/tensorrt-llm/release" in result.message
    assert "qwen-decode-66448fc968-jdgtp" in result.message


def test_the_right_launcher_from_the_wrong_image_fails():
    """Same engine, different build source: the image is its own fact."""
    engine, repo, command = ENGINES[0]
    result = BaseSmoketest.assert_engine_identity(
        _pod(command, "ghcr.io/someone/vllm-fork:dev"),
        engine,
        expected_repository=repo,
    )
    assert not result.passed
    assert "ghcr.io/someone/vllm-fork" in result.message


def test_a_pod_with_no_containers_does_not_claim_an_engine():
    result = BaseSmoketest.assert_engine_identity({"spec": {"containers": []}}, "vllm")
    assert not result.passed
    assert "no serving container" in result.message


def test_a_pod_of_nothing_but_sidecars_fails_on_the_image():
    """`_engine_container` falls back to the first container so a *renamed*
    engine container is still inspected. That fallback means the sidecar gets
    inspected here, and the image is what gives it away."""
    pod = {
        "spec": {
            "containers": [
                {
                    "name": "routing-proxy",
                    "image": "ghcr.io/llm-d/llm-d-routing-sidecar:v0.3.0",
                }
            ]
        }
    }
    result = BaseSmoketest.assert_engine_identity(
        pod, "vllm", expected_repository="docker.io/vllm/vllm-openai"
    )
    assert not result.passed
    assert "llm-d-routing-sidecar" in result.message


@pytest.mark.parametrize(
    "image",
    [
        "docker.io/vllm/vllm-openai@sha256:" + "a" * 64,  # digest-pinned
        "localhost:5000/vllm/vllm-openai:v1",  # registry port, not a tag
        "docker.io/vllm/vllm-openai",  # no tag at all
    ],
)
def test_image_reference_forms_all_resolve_to_their_repository(image):
    engine, _, command = ENGINES[0]
    expected = image.split("@")[0].rsplit(":", 1)[0] if "@" in image else None
    expected = expected or (
        "localhost:5000/vllm/vllm-openai"
        if image.startswith("localhost")
        else "docker.io/vllm/vllm-openai"
    )
    result = BaseSmoketest.assert_engine_identity(
        _pod(command, image), engine, expected_repository=expected
    )
    assert result.passed, result.message


@pytest.mark.parametrize("engine", ["generic", "someengine"])
def test_an_engine_with_no_launcher_signature_says_so_instead_of_claiming_one(engine):
    """`generic` is what `resolve_engines` writes for a command whose launcher it
    did not recognise -- a supported way to run a scenario, not a fault -- and it
    has no signature to match. The check must not fail those, nor imply it
    checked something it could not."""
    result = BaseSmoketest.assert_engine_identity(
        _pod("my-server --port 8200", "ghcr.io/me/server:v1"),
        engine,
        expected_repository="ghcr.io/me/server",
    )
    assert result.passed, result.message
    assert "launcher unchecked" in result.message
    # The name the plan stated, not the generic spec's, when they differ.
    assert engine in result.message


def test_an_alias_is_reported_under_its_engines_own_name():
    _, _, command = ENGINES[2]
    result = BaseSmoketest.assert_engine_identity(
        _pod(command, "nvcr.io/nvidia/tensorrt-llm/release:1.0"), "tensorrt-llm"
    )
    assert result.passed, result.message
    assert "'trtllm'" in result.message


def test_a_container_with_no_args_still_checks_the_image():
    pod = {
        "spec": {
            "containers": [
                {"name": "modelserver", "image": "docker.io/lmsysorg/sglang:v0.5.19"}
            ]
        }
    }
    result = BaseSmoketest.assert_engine_identity(
        pod, "sglang", expected_repository="docker.io/lmsysorg/sglang"
    )
    assert result.passed, result.message
    assert "launcher unchecked" in result.message
    assert "docker.io/lmsysorg/sglang" in result.message


def test_args_that_launch_nothing_known_fail_a_known_engine():
    result = BaseSmoketest.assert_engine_identity(
        _pod("sleep infinity", "docker.io/vllm/vllm-openai:v1"),
        "vllm",
        expected_repository="docker.io/vllm/vllm-openai",
    )
    assert not result.passed
    assert "vllm" in result.message
