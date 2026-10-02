"""Tests for the verbatim engine-command parser.

The scenario carries the engine launch command exactly as a user would type
it, and :mod:`llmdbenchmark.engine.command` reads back only the handful of
facts Kubernetes needs before the process starts: which engine, which model,
which bind port, and the handful of numbers something outside the engine has to
agree with -- the two the pre-deploy capacity check sizes KV cache against, and
the KV page size the router's prefix-cache index hashes on. Everything else is
opaque and must pass through untouched -- including the parallelism widths,
which the engine reads from the command like any other flag.

These tests pin that contract per engine, using command text copied from the
llm-d guides, so a new engine or a new flag spelling cannot quietly change
what the orchestrator infers.
"""

from __future__ import annotations

import pytest

from llmdbenchmark.engine import (
    ENGINE_SPECS,
    MODEL_READS,
    detect_engine,
    get_engine_spec,
    model_id_from_commands,
    parse_command,
    tokenize,
)


# ---------------------------------------------------------------------------
# launcher recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("vllm serve Qwen/Qwen3-0.6B", "vllm"),
        ("/usr/local/bin/vllm serve Qwen/Qwen3-0.6B", "vllm"),
        ("python3 -m vllm.entrypoints.openai.api_server --model x", "vllm"),
        ("sglang serve x", "sglang"),
        ("python3 -m sglang.launch_server --model-path x", "sglang"),
        ("trtllm-serve Qwen/Qwen3-0.6B", "trtllm"),
        ("trtllm-serve serve Qwen/Qwen3-0.6B", "trtllm"),
        ("/opt/venv/bin/trtllm-serve serve Qwen/Qwen3-0.6B", "trtllm"),
        ("llm-d-inference-sim --model x", "sim"),
    ],
)
def test_launcher_identifies_the_engine(text, expected):
    assert parse_command(text).engine == expected


def test_unrecognised_launcher_notes_instead_of_failing():
    """A snippet we cannot read is advisory, never an error.

    Common flags are still read with the generic spec so a stack can stand up;
    the note tells the user which facts cannot be inferred safely.
    """
    parsed = parse_command("/opt/app-root/spyre_entrypoint.sh --model x --port 8200")

    assert any("could not identify the engine launcher" in n for n in parsed.notes)
    assert parsed.engine == "generic"
    assert parsed.port == 8200


def test_vllm_benchmark_client_is_not_mistaken_for_a_server():
    parsed = parse_command("vllm bench serve --backend openai --port 8000")

    assert parsed.engine == "generic"
    assert any("could not identify the engine launcher" in n for n in parsed.notes)


def test_declared_engine_reads_an_unrecognised_wrapper():
    """A wrapper script that execs the engine is read in the declared spelling.

    This is what ``engine.name`` exists for -- the Spyre images launch vLLM
    through ``/opt/app-root/spyre_entrypoint.sh``, which matches no launcher
    signature, so the scenario declares the engine and the flags become
    readable.
    """
    text = (
        "/opt/app-root/spyre_entrypoint.sh --model Qwen/Qwen3-32B "
        "--port 8200 --tensor-parallel-size 4 --max-model-len 32768"
    )
    parsed = parse_command(text, engine="vllm")

    assert parsed.engine == "vllm"
    assert parsed.port == 8200
    assert parsed.reads["maxModelLen"] == 32768
    assert not any("could not identify" in n for n in parsed.notes)


# ---------------------------------------------------------------------------
# subcommands of a launcher that is a command group
# ---------------------------------------------------------------------------


def test_trtllm_subcommand_is_not_mistaken_for_the_model():
    """``trtllm-serve`` is a click group whose unrecognised first argument
    falls through to ``serve``, so both spellings name one invocation and must
    parse identically."""
    bare = parse_command("trtllm-serve Qwen/Qwen3-0.6B --port 8200")
    grouped = parse_command("trtllm-serve serve Qwen/Qwen3-0.6B --port 8200")

    assert bare.model == "Qwen/Qwen3-0.6B"
    assert grouped.model == "Qwen/Qwen3-0.6B"
    assert bare.port == grouped.port == 8200


def test_only_a_declared_subcommand_is_skipped():
    """A model whose name happens to lead the positionals is still the model."""
    parsed = parse_command("trtllm-serve serve serve-me/Model-1B")

    assert parsed.model == "serve-me/Model-1B"


def test_vllm_serve_is_a_launcher_not_a_subcommand():
    """vLLM's ``serve`` is part of the launcher signature, so the positional
    after it is the model -- the subcommand strip must not double-consume."""
    parsed = parse_command("vllm serve Qwen/Qwen3-0.6B")

    assert parsed.model == "Qwen/Qwen3-0.6B"
    assert get_engine_spec("vllm").subcommands == ()


# ---------------------------------------------------------------------------
# per-engine flag spellings
# ---------------------------------------------------------------------------


def test_vllm_command_from_the_guide():
    parsed = parse_command(
        """
        vllm serve Qwen/Qwen3-32B \
        --host 0.0.0.0 \
        --port 8200 \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --block-size 64 \
        --gpu-memory-utilization 0.95 \
        --max-num-seqs 256 \
        --max-num-batched-tokens 8192
        """
    )

    assert parsed.engine == "vllm"
    # The serve target is the model id, written once. vLLM advertises whatever
    # it was asked to serve, so there is no --served-model-name to agree with.
    assert parsed.model == "Qwen/Qwen3-32B"
    assert parsed.servedModelName is None
    assert parsed.port == 8200
    assert parsed.reads == {
        "maxModelLen": 32768,
        "gpuMemoryUtilization": 0.95,
        "blockSize": 64,
    }
    # Everything else the command says is the engine's business. It is present
    # in the raw text and readable, but nothing is derived from it.
    assert parsed.flags["--tensor-parallel-size"] == "4"
    assert parsed.flags["--max-num-seqs"] == "256"


def test_sglang_command_from_the_guide():
    """SGLang names every one of these differently from vLLM, and the scenario
    restates none of them -- ``--page-size`` is the same fact as vLLM's
    ``--block-size`` and lands on the same key."""
    parsed = parse_command(
        """
        sglang serve Qwen/Qwen3-0.6B \
        --host 0.0.0.0 \
        --port 8200 \
        --tp-size 4 \
        --context-length 32768 \
        --page-size 64 \
        --mem-fraction-static 0.9 \
        --max-running-requests 256 \
        --max-prefill-tokens 8192
        """
    )

    assert parsed.engine == "sglang"
    assert parsed.model == "Qwen/Qwen3-0.6B"
    assert parsed.servedModelName is None
    assert parsed.port == 8200
    assert parsed.reads == {
        "maxModelLen": 32768,
        "gpuMemoryUtilization": 0.9,
        "blockSize": 64,
    }
    assert parsed.flags["--tp-size"] == "4"


def test_sglang_model_flag_alias_is_read():
    """SGLang declares ``--model-path`` with ``--model`` as an alias, so a
    command written either way names the same model."""
    assert parse_command("python3 -m sglang.launch_server --model m/M").model == "m/M"
    assert (
        parse_command("python3 -m sglang.launch_server --model-path m/M").model == "m/M"
    )
    assert parse_command("sglang serve m/M").model == "m/M"


def test_trtllm_command_from_the_guide():
    """TRT-LLM spells its flags with underscores. The capacity pair is read in
    that spelling; the rest passes through as written.

    No ``blockSize``: TRT-LLM has no CLI flag for its KV page size at all --
    ``tokens_per_block`` lives inside the ``--extra_llm_api_options`` YAML, so a
    scenario that needs the number elsewhere states ``model.blockSize`` and
    nothing is read back.
    """
    parsed = parse_command(
        """
        trtllm-serve serve Qwen/Qwen3-0.6B \
        --host 0.0.0.0 \
        --port 8200 \
        --backend pytorch \
        --tp_size 2 \
        --max_seq_len 32768 \
        --max_batch_size 256 \
        --max_num_tokens 8192 \
        --free_gpu_memory_fraction 0.9 \
        --extra_llm_api_options /tmp/trtllm/llm_api_options.yaml
        """
    )

    assert parsed.engine == "trtllm"
    assert parsed.model == "Qwen/Qwen3-0.6B"
    assert parsed.servedModelName is None
    assert parsed.port == 8200
    assert parsed.reads == {
        "maxModelLen": 32768,
        "gpuMemoryUtilization": 0.9,
    }
    assert "blockSize" not in parsed.reads
    assert get_engine_spec("trtllm").block_size_flags == ()
    assert parsed.flags["--tp_size"] == "2"
    assert parsed.flags["--extra_llm_api_options"] == "/tmp/trtllm/llm_api_options.yaml"


def test_trtllm_memory_fraction_is_read_in_either_spelling():
    """``trtllm-serve`` declares one option under two names.

    ``@stability_option("--free_gpu_memory_fraction",
    "--kv_cache_free_gpu_memory_fraction", ...)`` means the engine answers to
    both, and the llm-d optimized-baseline TRT-LLM guide writes the longer one
    (guides/optimized-baseline/modelserver/gpu/trtllm/patch-trtllm.yaml). A user
    pasting that line must get the same capacity check as a user pasting the
    short spelling -- reading only one of the two would skip it silently.
    """
    base = "trtllm-serve serve Qwen/Qwen3-0.6B --port 8200 --max_seq_len 32768 "

    short = parse_command(base + "--free_gpu_memory_fraction 0.9")
    long = parse_command(base + "--kv_cache_free_gpu_memory_fraction 0.9")

    assert short.reads["gpuMemoryUtilization"] == 0.9
    assert long.reads["gpuMemoryUtilization"] == 0.9


# ---------------------------------------------------------------------------
# what is deliberately not inferred
# ---------------------------------------------------------------------------


def test_parallelism_widths_are_not_interpreted():
    """A device count is a Kubernetes fact, not a reading of the command.

    The kubelet grants devices before the engine process exists, so the count
    comes from the pod spec (``resources.limits.<accelerator resource>``, or the
    ``accelerator.count`` shorthand) and the parallelism the llm-d chart needs
    is stated as the chart value it is. Inferring it from a product of flag
    widths would mean tracking every engine's spelling of every width -- and
    would disagree with the pod spec the moment one of them changed.
    """
    parsed = parse_command(
        "vllm serve m --tensor-parallel-size 2 --pipeline-parallel-size 2 "
        "--data-parallel-size 8 --data-parallel-size-local 2"
    )

    assert not hasattr(parsed, "parallelism")
    assert not hasattr(parsed, "accelerator_count")
    # Read, yes -- so a warning can quote it -- but nothing is derived from it.
    assert parsed.flags["--data-parallel-size-local"] == "2"


@pytest.mark.parametrize(
    "flag",
    [
        "--max-num-seqs 256",
        "--max-num-batched-tokens 8192",
        "--enable-expert-parallel",
        "--tensor-parallel-size 4",
        "--dtype bfloat16",
        '--kv-transfer-config {"kv_connector":"NixlConnector"}',
    ],
)
def test_engine_flags_beyond_the_reads_stay_opaque(flag):
    """``MODEL_READS`` is the whole list, and it is short because each entry
    exists for something outside the engine that would otherwise have to be told
    the same number twice. Every other flag reaches the container verbatim and
    has no key anywhere."""
    parsed = parse_command(f"vllm serve m --port 8200 {flag}")

    assert parsed.reads == {}
    assert set(parsed.reads) <= set(MODEL_READS)
    assert flag.split()[0] in parsed.flags
    assert flag.split()[0] in parsed.raw


@pytest.mark.parametrize(
    "command,expected",
    [
        ("vllm serve m --block-size 64", 64),
        ("python3 -m sglang.launch_server --model-path m --page-size 32", 32),
        ("llm-d-inference-sim --model m --block-size 16", 16),
    ],
)
def test_kv_page_size_is_read_in_each_engines_spelling(command, expected):
    """The KV page size is read because something outside the engine hashes on
    it: the router's prefix-cache token processor reconstructs block hashes on
    the engine's own boundaries, and a number that disagrees scores silently
    against the wrong blocks. Each engine spells the flag differently and the
    scenario writes only the engine's spelling."""
    assert parse_command(command).reads["blockSize"] == expected


# ---------------------------------------------------------------------------
# shell snippets: preamble, env prefixes, substitution placeholders
# ---------------------------------------------------------------------------


def test_preamble_is_separated_from_the_launch():
    parsed = parse_command(
        "export VLLM_LOGGING_LEVEL=DEBUG ; "
        "source /setup/env.sh && "
        "vllm serve m --port 8200 --tensor-parallel-size 2"
    )

    assert "source /setup/env.sh" in parsed.preamble
    assert "vllm serve" not in parsed.preamble
    assert parsed.port == 8200


def test_env_prefix_on_the_launch_itself_is_skipped():
    parsed = parse_command(
        "VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0,1 vllm serve m --port 8200"
    )

    assert parsed.engine == "vllm"
    assert parsed.model == "m"
    assert parsed.port == 8200


def test_shell_variables_survive_parsing():
    """``$MODEL_NAME`` and ``${LWS_WORKER_INDEX:-0}`` are the container's to
    expand, not ours -- the raw text is what reaches the engine."""
    text = (
        "vllm serve $MODEL_NAME --port 8200 "
        "--data-parallel-start-rank ${LWS_WORKER_INDEX:-0}"
    )
    parsed = parse_command(text)

    assert parsed.model == "$MODEL_NAME"
    assert parsed.raw == text
    assert "${LWS_WORKER_INDEX:-0}" in parsed.raw


def test_empty_command_parses_to_nothing():
    for text in (None, "", "   \n  "):
        parsed = parse_command(text)
        assert parsed.engine is None
        assert parsed.port is None
        assert parsed.reads == {}
        assert parsed.notes == []


# ---------------------------------------------------------------------------
# registry invariants
# ---------------------------------------------------------------------------


def test_every_spec_has_a_launcher_and_a_default_port():
    """Both are load-bearing: without a launcher signature a command cannot be
    recognised, and without a default port a role that runs the image's own
    entrypoint has no port to put on the Service."""
    for spec in ENGINE_SPECS:
        assert spec.launchers, f"{spec.name} has no launcher signature"
        assert spec.default_port, f"{spec.name} has no default port"


def test_every_spec_is_reachable_by_name():
    for spec in ENGINE_SPECS:
        assert get_engine_spec(spec.name) is spec


def test_detect_engine_agrees_with_parse_command():
    for text in (
        "vllm serve m",
        "sglang serve m",
        "trtllm-serve serve m",
        "llm-d-inference-sim --model m",
    ):
        tokens, _ = tokenize(text)
        detected = detect_engine(tokens)
        assert detected is not None
        assert detected.name == parse_command(text).engine


def test_inline_comment_and_unspaced_separator_do_not_override_launch_flags():
    parsed = parse_command("vllm serve model --port 8200; echo done # --port 9000")

    assert parsed.port == 8200


# ---------------------------------------------------------------------------
# what happens when a read and a stated value disagree
# ---------------------------------------------------------------------------


def _resolve(values):
    from llmdbenchmark.engine import resolve_engines

    return resolve_engines(values), values


def test_a_read_fills_a_model_number_the_scenario_left_unset():
    warnings, values = _resolve(
        {
            "model": {},
            "decode": {
                "engine": {"command": "vllm serve m --port 8200 --block-size 64"}
            },
        }
    )

    assert values["model"]["blockSize"] == 64
    assert warnings == []


def test_command_port_overrules_a_conflicting_explicit_port():
    warnings, values = _resolve(
        {
            "model": {},
            "decode": {
                "engine": {
                    "command": "vllm serve m --port 8200",
                    "port": 8000,
                }
            },
        }
    )

    assert values["decode"]["engine"]["port"] == 8200
    assert len(warnings) == 1
    assert "command binds 8200" in warnings[0]
    assert "command is authoritative" in warnings[0]


def test_declared_custom_engine_name_is_preserved():
    warnings, values = _resolve(
        {
            "model": {},
            "decode": {
                "engine": {
                    "name": "my-engine",
                    "command": "my-engine serve --model m --port 9000",
                    "image": {"repository": "example/my-engine", "tag": "v1"},
                }
            },
        }
    )

    assert values["decode"]["engine"]["name"] == "my-engine"
    assert values["resolvedServingRoles"]["decode"]["engineName"] == "my-engine"
    assert values["decode"]["engine"]["port"] == 9000
    assert any(
        "could not identify the engine launcher" in warning for warning in warnings
    )


def test_disabled_standalone_does_not_warn_about_inherited_disaggregation_flags():
    warnings, _ = _resolve(
        {
            "model": {},
            "decode": {
                "engine": {
                    "command": (
                        "vllm serve m --port 8200 "
                        '--kv-transfer-config \'{"kv_connector":"NixlConnector"}\''
                    )
                }
            },
            "standalone": {"enabled": False, "engine": {}},
        }
    )

    assert not any(
        "standalone has no engine.command" in warning for warning in warnings
    )


def test_active_standalone_warns_about_inherited_disaggregation_flags():
    warnings, _ = _resolve(
        {
            "model": {},
            "decode": {
                "engine": {
                    "command": (
                        "vllm serve m --port 8200 "
                        '--kv-transfer-config \'{"kv_connector":"NixlConnector"}\''
                    )
                }
            },
            "standalone": {"enabled": True, "engine": {}},
        }
    )

    assert any("standalone has no engine.command" in warning for warning in warnings)


def test_a_read_overrules_a_stated_number_and_says_so():
    """The command is the text the engine is handed, so it decides.

    A value stated elsewhere -- a scenario's own ``model:`` block, an accelerator
    overlay that knows the kernel picks its own page size -- is a fallback for a
    command that says nothing. When the command does say, a consumer told the
    other number would be hashing pages the engine never writes, so the command
    wins and the override is reported.
    """
    warnings, values = _resolve(
        {
            "model": {"blockSize": 16},
            "decode": {
                "engine": {"command": "vllm serve m --port 8200 --block-size 64"}
            },
        }
    )

    assert values["model"]["blockSize"] == 64
    assert len(warnings) == 1
    assert "model.blockSize was set to 16" in warnings[0]
    assert "--block-size 64" in warnings[0]


def test_a_stated_number_the_command_cannot_carry_is_kept_silently():
    """TRT-LLM has no CLI flag for its KV page size at all, so there is nothing
    to read and nothing to disagree with."""
    warnings, values = _resolve(
        {
            "model": {"blockSize": 32},
            "decode": {
                "engine": {"command": "trtllm-serve serve m --port 8200"},
            },
        }
    )

    assert values["model"]["blockSize"] == 32
    assert warnings == []


def test_a_disabled_roles_command_decides_nothing():
    """``prefill`` is present in every scenario and off in most; its command
    must not set the plan's numbers."""
    _, values = _resolve(
        {
            "model": {},
            "prefill": {
                "enabled": False,
                "engine": {"command": "vllm serve m --port 8000 --block-size 128"},
            },
            "decode": {
                "engine": {"command": "vllm serve m --port 8200 --block-size 64"}
            },
        }
    )

    assert values["model"]["blockSize"] == 64


def test_inactive_modelservice_command_decides_nothing_in_standalone_mode():
    values = {
        "model": {},
        "modelservice": {"enabled": False},
        "decode": {
            "enabled": True,
            "engine": {
                "command": "vllm serve decode-model --port 8200 --block-size 64"
            },
        },
        "standalone": {
            "enabled": True,
            "engine": {
                "command": "sglang serve standalone-model --port 30000 --page-size 32"
            },
        },
    }

    assert model_id_from_commands(values) == "standalone-model"
    _, resolved = _resolve(values)
    assert resolved["model"]["blockSize"] == 32


def test_standalone_inherits_the_model_id_from_decodes_command():
    values = {
        "model": {},
        "modelservice": {"enabled": False},
        "decode": {
            "engine": {
                "command": "vllm serve Qwen/Qwen3-0.6B --port 8200",
                "extraArgs": ["--served-model-name", "qwen"],
            },
        },
        "standalone": {
            "enabled": True,
            "engine": {"extraArgs": ["--dtype", "bfloat16"]},
        },
    }

    model_id = model_id_from_commands(values)
    assert model_id == "qwen"
    values["model"]["name"] = model_id
    warnings, resolved = _resolve(values)
    standalone = resolved["resolvedServingRoles"]["standalone"]
    assert standalone["servedModelName"] == "qwen"
    assert standalone["command"].endswith("--dtype bfloat16")
    assert not any("model.name" in warning for warning in warnings)


def test_standalone_inherits_decodes_command_runtime_settings():
    values = {
        "model": {"name": "model"},
        "modelservice": {"enabled": False},
        "images": {"vllm": {"repository": "default/vllm", "tag": "latest"}},
        "decode": {
            "engine": {
                "name": "my-engine",
                "command": "my-engine serve --model model --port $ENGINE_PORT",
                "port": 9000,
                "preprocessCommand": "prepare-engine",
                "image": {
                    "repository": "example/my-engine",
                    "tag": "v1",
                    "pullPolicy": "Always",
                },
                "healthPath": "/ready",
                "metricsPath": "/prometheus",
                "containerName": "custom-server",
            },
        },
        "standalone": {"enabled": True, "engine": {}},
    }

    _, resolved = _resolve(values)
    standalone = resolved["standalone"]["engine"]

    assert standalone["name"] == "my-engine"
    assert standalone["port"] == 9000
    assert standalone["preprocessCommand"] == "prepare-engine"
    assert standalone["image"] == {
        "repository": "example/my-engine",
        "tag": "v1",
        "pullPolicy": "Always",
    }
    assert standalone["healthPath"] == "/ready"
    assert standalone["metricsPath"] == "/prometheus"
    assert standalone["containerName"] == "custom-server"


def test_empty_standalone_command_does_not_inherit_a_model_id():
    values = {
        "model": {},
        "modelservice": {"enabled": False},
        "decode": {"engine": {"command": "vllm serve decode-model"}},
        "standalone": {"enabled": True, "engine": {"command": ""}},
    }

    assert model_id_from_commands(values) is None
