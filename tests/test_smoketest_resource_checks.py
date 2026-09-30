"""``assert_resource_matches`` reads a resource path, not a dotted tree.

A container's resource section is a flat map keyed by resource *name*, and a
Kubernetes extended resource name contains dots: ``nvidia.com/gpu``,
``habana.ai/gaudi``, ``amd.com/gpu``. Splitting ``limits.nvidia.com/gpu`` on
every dot looks for ``limits`` -> ``nvidia`` -> ``com/gpu``, finds nothing, and
reports the accelerator as unset on a pod that has its devices -- which is how
four scenarios failed their GPU check while serving traffic.

Only the first dot separates the section from the name.
"""

from __future__ import annotations

import pytest

from llmdbenchmark.smoketests.base import BaseSmoketest


ACCELERATORS = ("nvidia.com/gpu", "amd.com/gpu", "habana.ai/gaudi", "google.com/tpu")


@pytest.mark.parametrize("resource", ACCELERATORS)
def test_extended_resource_name_keeps_its_dots(resource):
    result = BaseSmoketest.assert_resource_matches(
        {"limits": {resource: "2"}, "requests": {resource: "2"}},
        "2",
        f"limits.{resource}",
    )
    assert result.passed, result.message
    assert result.name == f"resource_limits.{resource}"


def test_plain_resource_name_still_reads():
    result = BaseSmoketest.assert_resource_matches(
        {"limits": {"memory": "64Gi"}}, "64Gi", "limits.memory"
    )
    assert result.passed, result.message


def test_mismatch_reports_both_values():
    result = BaseSmoketest.assert_resource_matches(
        {"limits": {"nvidia.com/gpu": "1"}}, "2", "limits.nvidia.com/gpu"
    )
    assert not result.passed
    assert result.actual == "1"
    assert result.expected == "2"


@pytest.mark.parametrize(
    "resources",
    [
        {},  # no resources at all
        {"limits": {}},  # a section with nothing in it
        {"requests": {"nvidia.com/gpu": "1"}},  # set, but in the other section
        {"limits": "1"},  # not a mapping
    ],
)
def test_absent_value_reads_as_not_set(resources):
    """'not set' has to mean absent -- never a stringified default."""
    result = BaseSmoketest.assert_resource_matches(
        resources, "1", "limits.nvidia.com/gpu"
    )
    assert not result.passed
    assert result.actual == "not set"
    assert "{}" not in result.message
