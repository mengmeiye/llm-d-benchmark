"""Tests for the pydantic config validation schema.

Validates that:
- defaults.yaml passes validation without errors
- Example scenario files (merged with defaults) pass validation
- Typos in modeled sections are caught
- Type errors are caught
- Constraint violations are caught
- validate_config() never raises exceptions (non-blocking)
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from llmdbenchmark.parser.config_schema import (
    BenchmarkConfig,
    validate_config,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = PROJECT_ROOT / "config" / "templates" / "values" / "defaults.yaml"
SCENARIOS_EXAMPLES = PROJECT_ROOT / "config" / "scenarios" / "examples"
SCENARIOS_GUIDES = PROJECT_ROOT / "config" / "scenarios" / "guides"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def defaults() -> dict:
    """Load defaults.yaml once for all tests."""
    return _load_yaml(DEFAULTS_PATH)


@pytest.fixture
def defaults_copy(defaults: dict) -> dict:
    """Return a deep copy of defaults for mutation tests."""
    return copy.deepcopy(defaults)


# ---------------------------------------------------------------------------
# Test: defaults.yaml passes validation
# ---------------------------------------------------------------------------


class TestDefaultsValidation:
    """defaults.yaml should pass validation without any warnings."""

    def test_defaults_pass_validation(self, defaults: dict) -> None:
        warnings = validate_config(defaults)
        assert warnings == [], f"Unexpected warnings: {warnings}"

    def test_defaults_produce_valid_model(self, defaults: dict) -> None:
        config = BenchmarkConfig.model_validate(defaults)
        assert config.model.name == "facebook/opt-125m"
        assert config.decode.enabled is True
        assert config.prefill.enabled is False
        assert config.engine.servicePort == 8000
        # llm-d's engine-neutral container name: no manifest, probe or log
        # lookup should have to know which engine is running.
        assert config.engine.containerName == "modelserver"


# ---------------------------------------------------------------------------
# Test: example scenarios pass validation when merged with defaults
# ---------------------------------------------------------------------------


class TestScenarioValidation:
    """Merged defaults + scenario should pass validation."""

    @staticmethod
    def _scenario_files() -> list[Path]:
        files = []
        for directory in [SCENARIOS_EXAMPLES, SCENARIOS_GUIDES]:
            if directory.exists():
                files.extend(sorted(directory.glob("*.yaml")))
        return files

    @pytest.mark.parametrize(
        "scenario_path",
        _scenario_files.__func__(),
        ids=lambda p: p.name,
    )
    def test_scenario_passes_validation(
        self, defaults: dict, scenario_path: Path
    ) -> None:
        scenario = _load_yaml(scenario_path)
        merged = _deep_merge(defaults, scenario)
        warnings = validate_config(merged)
        assert warnings == [], (
            f"Scenario {scenario_path.name} produced warnings:\n" + "\n".join(warnings)
        )


# ---------------------------------------------------------------------------
# Test: typo detection (extra="forbid" catches unknown keys)
# ---------------------------------------------------------------------------


class TestTypoDetection:
    """Misspelled keys in modeled sections should be caught.

    The misspellings below are DELIBERATE -- each test feeds an unknown key
    to the schema and asserts ``extra="forbid"`` rejects it. Correcting one
    to its real spelling makes the key legitimate, so validation returns no
    warnings and the test fails. Do not let a spell-checker "fix" them; the
    sentinels are allow-listed in ``_typos.toml``.
    """

    def test_decode_typo_caught(self, defaults_copy: dict) -> None:
        defaults_copy["decode"]["replicsa"] = 2
        warnings = validate_config(defaults_copy)
        assert any("replicsa" in w for w in warnings), (
            f"Expected 'replicsa' typo to be caught, got: {warnings}"
        )

    def test_model_typo_caught(self, defaults_copy: dict) -> None:
        defaults_copy["model"]["naem"] = "test"
        warnings = validate_config(defaults_copy)
        assert any("naem" in w for w in warnings), (
            f"Expected 'naem' typo to be caught, got: {warnings}"
        )

    def test_engine_common_typo_caught(self, defaults_copy: dict) -> None:
        defaults_copy["engine"]["serviceProt"] = 8000
        warnings = validate_config(defaults_copy)
        assert any("serviceProt" in w for w in warnings), (
            f"Expected 'serviceProt' typo to be caught, got: {warnings}"
        )

    def test_harness_typo_caught(self, defaults_copy: dict) -> None:
        defaults_copy["harness"]["waitTimout"] = 3600
        warnings = validate_config(defaults_copy)
        assert any("waitTimout" in w for w in warnings), (
            f"Expected 'waitTimout' typo to be caught, got: {warnings}"
        )

    def test_prefill_engine_typo_caught(self, defaults_copy: dict) -> None:
        # The engine section is the one place a scenario now writes, so a typo
        # here is the most likely one there is -- and a silently ignored
        # `comand` would stand the role up on the image's entrypoint instead.
        defaults_copy["prefill"]["engine"]["comand"] = "vllm serve m"
        warnings = validate_config(defaults_copy)
        assert any("comand" in w for w in warnings), (
            f"Expected 'comand' typo to be caught, got: {warnings}"
        )


# ---------------------------------------------------------------------------
# Test: type error detection
# ---------------------------------------------------------------------------


class TestTypeErrors:
    """Wrong types in modeled sections should be caught."""

    def test_model_gpu_util_out_of_range(self, defaults_copy: dict) -> None:
        defaults_copy["model"]["gpuMemoryUtilization"] = 2.0
        warnings = validate_config(defaults_copy)
        assert len(warnings) > 0, (
            "Expected constraint violation for gpuMemoryUtilization > 1"
        )

    def test_decode_replicas_negative(self, defaults_copy: dict) -> None:
        defaults_copy["decode"]["replicas"] = -1
        warnings = validate_config(defaults_copy)
        assert len(warnings) > 0, "Expected constraint violation for negative replicas"

    def test_harness_wait_timeout_negative(self, defaults_copy: dict) -> None:
        defaults_copy["harness"]["waitTimeout"] = -100
        warnings = validate_config(defaults_copy)
        assert len(warnings) > 0, (
            "Expected constraint violation for negative waitTimeout"
        )


# ---------------------------------------------------------------------------
# Test: non-blocking behavior
# ---------------------------------------------------------------------------


class TestNonBlocking:
    """validate_config() must never raise — always return a list."""

    def test_returns_list_on_valid(self, defaults: dict) -> None:
        result = validate_config(defaults)
        assert isinstance(result, list)

    def test_returns_list_on_invalid(self, defaults_copy: dict) -> None:
        defaults_copy["model"]["naem"] = "bad"
        result = validate_config(defaults_copy)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_returns_list_on_garbage_input(self) -> None:
        result = validate_config({"model": "not-a-dict"})
        assert isinstance(result, list)
        assert len(result) > 0

    def test_returns_list_on_empty_input(self) -> None:
        result = validate_config({})
        assert isinstance(result, list)
        assert len(result) > 0  # empty dict is missing required sections


# ---------------------------------------------------------------------------
# Test: extra="allow" sections accept arbitrary keys
# ---------------------------------------------------------------------------


class TestAllowSections:
    """Sections with extra="allow" should not reject unknown keys."""

    def test_resource_limits_accept_gpu_key(self, defaults_copy: dict) -> None:
        defaults_copy["decode"]["resources"]["limits"]["nvidia.com/gpu"] = "1"
        warnings = validate_config(defaults_copy)
        assert warnings == [], f"GPU resource key should be accepted: {warnings}"

    def test_extra_container_config_accepts_arbitrary(
        self, defaults_copy: dict
    ) -> None:
        defaults_copy["decode"]["extraContainerConfig"]["securityContext"] = {
            "privileged": True
        }
        warnings = validate_config(defaults_copy)
        assert warnings == [], (
            f"extraContainerConfig should accept arbitrary keys: {warnings}"
        )

    def test_engine_command_accepts_unknown_flags(self, defaults_copy: dict) -> None:
        """A flag llm-d-benchmark has never heard of must still validate.

        This is the whole point of the verbatim-command contract: engine flags
        live in the command string, which the schema does not interpret, so a
        new vLLM/SGLang release needs no change here.
        """
        defaults_copy["decode"]["engine"]["command"] = (
            "vllm serve ${model.name} --port 8200 --some-flag-from-next-release 7"
        )
        warnings = validate_config(defaults_copy)
        assert warnings == [], f"engine.command should not be parsed: {warnings}"

    def test_root_accepts_unknown_top_level(self, defaults_copy: dict) -> None:
        defaults_copy["someNewTopLevelKey"] = {"foo": "bar"}
        warnings = validate_config(defaults_copy)
        assert warnings == [], (
            f"Root model should accept unknown top-level keys: {warnings}"
        )


# ---------------------------------------------------------------------------
# Test: scenario-only fields are accepted
# ---------------------------------------------------------------------------


class TestScenarioOnlyFields:
    """Fields that exist in scenarios but not defaults must be accepted."""

    def test_engine_tensor_parallelism_rejected(self, defaults_copy: dict) -> None:
        # Parallelism is stated once, in the command (`--tensor-parallel-size`
        # / `--tp-size` / ...), and read back out by resolve_engines. A second
        # place to say it could disagree with the command, so it is not a key.
        defaults_copy["engine"]["tensorParallelism"] = 4
        warnings = validate_config(defaults_copy)
        assert any("tensorParallelism" in w for w in warnings), (
            "engine.tensorParallelism should be rejected (state it in the command)"
        )

    def test_engine_max_model_len_rejected(self, defaults_copy: dict) -> None:
        # Same reasoning: context length is a pure engine parameter and belongs
        # only in the command.
        defaults_copy["engine"]["maxModelLen"] = 8192
        warnings = validate_config(defaults_copy)
        assert any("maxModelLen" in w for w in warnings), (
            "engine.maxModelLen should be rejected (state it in the command)"
        )

    def test_engine_shm_memory(self, defaults_copy: dict) -> None:
        # Not an engine parameter: it sizes the /dev/shm volume on the pod, so
        # it stays a key.
        defaults_copy["engine"]["shmMemory"] = "16Gi"
        warnings = validate_config(defaults_copy)
        assert warnings == [], f"engine.shmMemory should be accepted: {warnings}"

    def test_harness_experiment_profile(self, defaults_copy: dict) -> None:
        defaults_copy["harness"]["experimentProfile"] = "sanity_random.yaml"
        warnings = validate_config(defaults_copy)
        assert warnings == [], (
            f"harness.experimentProfile should be accepted: {warnings}"
        )

    def test_work_dir(self, defaults_copy: dict) -> None:
        defaults_copy["workDir"] = "/workspace"
        warnings = validate_config(defaults_copy)
        assert warnings == [], f"workDir should be accepted: {warnings}"

    def test_decode_accelerator(self, defaults_copy: dict) -> None:
        defaults_copy["decode"]["accelerator"] = {"count": 0}
        warnings = validate_config(defaults_copy)
        assert warnings == [], f"decode.accelerator should be accepted: {warnings}"

    def test_decode_engine_model_command(self, defaults_copy: dict) -> None:
        defaults_copy["decode"]["engine"]["modelCommand"] = "imageDefault"
        warnings = validate_config(defaults_copy)
        assert warnings == [], (
            f"decode.engine.modelCommand should be accepted: {warnings}"
        )

    def test_a_batch_width_has_no_key(self, defaults_copy: dict) -> None:
        """Only the capacity pair is modeled under ``model``.

        ``--max-num-seqs`` is read by the engine and by nobody else, so there is
        no ``model.maxNumSeq`` for it to be restated in -- writing one is a typo,
        not a setting.
        """
        defaults_copy["model"]["maxNumSeq"] = 128
        warnings = validate_config(defaults_copy)
        assert any("maxNumSeq" in w for w in warnings), (
            f"Expected model.maxNumSeq to be rejected, got: {warnings}"
        )

    def test_no_prefix_caching_lives_in_the_command(self, defaults_copy: dict) -> None:
        """An engine flag is command text, not a config key."""
        defaults_copy["decode"]["engine"]["command"] = (
            "vllm serve ${model.name} --port 8200 --no-enable-prefix-caching"
        )
        warnings = validate_config(defaults_copy)
        assert warnings == [], f"Expected no warnings, got: {warnings}"
