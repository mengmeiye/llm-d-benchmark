"""Per-scenario validator registry.

Every scenario with a dedicated validator gets scenario-specific config
validation (step 2).  When a stack name is not found in the registry,
``get_validator()`` falls back to ``BaseSmoketest`` which still runs
generic health checks and inference tests (steps 0 and 1) -- it just
skips scenario-specific config validation.

The keys are **stack names** -- the ``scenario[].name`` a scenario file
declares, not its filename. A key that matches no stack is a validator that
never runs, and the fallback makes that silent: the smoketest passes, having
checked nothing scenario-specific. ``tests/test_validator_registry.py`` is what
makes it loud, so keep ``_SYNTHETIC_STACK_NAMES`` below honest.
"""

# Guides (well-lit paths)
from llmdbenchmark.smoketests.validators.pd_disaggregation import (
    PdDisaggregationValidator,
)
from llmdbenchmark.smoketests.validators.precise_prefix_cache_aware import (
    PrecisePrefixCacheAwareValidator,
)
from llmdbenchmark.smoketests.validators.optimized_baseline import (
    OptimizedBaselineValidator,
)
from llmdbenchmark.smoketests.validators.tiered_prefix_cache import (
    TieredPrefixCacheValidator,
)
from llmdbenchmark.smoketests.validators.wide_ep import WideEpValidator
from llmdbenchmark.smoketests.validators.wva import WvaValidator

# Examples
from llmdbenchmark.smoketests.validators.cpu import CpuValidator
from llmdbenchmark.smoketests.validators.gpu import GpuValidator
from llmdbenchmark.smoketests.validators.spyre import SpyreValidator
from llmdbenchmark.smoketests.validators.fma import FmaValidator


VALIDATORS: dict[str, type] = {
    # Guides (well-lit paths)
    "pd-disaggregation": PdDisaggregationValidator,
    "precise-prefix-cache-routing": PrecisePrefixCacheAwareValidator,
    "optimized-baseline": OptimizedBaselineValidator,
    # All FMA standup paths resolve here:
    # 1. guide path (standup_method: kustomize) has the
    # the guide stack name, and
    # 2.benchmark path (standup_method: fma) is mapped to
    # "fast-model-actuation" by get_validator.
    "fast-model-actuation": FmaValidator,
    "fast-model-actuation-base": FmaValidator,
    "fast-model-actuation-keda": FmaValidator,
    # workload-autoscaling deploys the same stack as optimized-baseline, so
    # it reuses that validator; the WvaSmoketestMixin auto-activates its
    # extra checks when the stack's config has wva.enabled: true.
    "workload-autoscaling": OptimizedBaselineValidator,
    "epp-keda-saturation": OptimizedBaselineValidator,
    "tiered-prefix-cache": TieredPrefixCacheValidator,
    "wide-ep": WideEpValidator,
    "wva": WvaValidator,
    # Examples
    "cpu-example": CpuValidator,
    "gpu-example": GpuValidator,
    "spyre-example": SpyreValidator,
}

# Registry keys that deliberately match no scenario's stack name.
#
# Anything not listed here and not claimed by a scenario is the silent-fallback
# bug: a validator wired to a name nothing deploys.
_SYNTHETIC_STACK_NAMES: frozenset[str] = frozenset(
    {
        # get_validator(is_fma=True) rewrites the stack name to this, so the
        # benchmark FMA path and the kustomize guide path share one validator.
        "fast-model-actuation",
        # No scenario ships this yet. Left registered on purpose so the
        # validator does not rot while the scenario is decided.
        "wva",
    }
)
