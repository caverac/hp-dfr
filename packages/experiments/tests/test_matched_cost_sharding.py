"""Unit tests for the matched-cost sweep's sharding logic.

These cover the partition and path helpers only, which are pure and cheap. The
sweep itself trains networks and is exercised on the cluster, not here.
"""

from __future__ import annotations

import pytest

from hp_dfr.data import (
    _DFR_STEP_MULT,
    _MATCHED_COST_SWEEPS,
    _matched_cost_path,
    _shard_items,
)


def test_shards_partition_every_item_exactly_once() -> None:
    """Every work item lands in exactly one shard, for any shard count."""
    items = list(range(37))  # a prime, so no count divides it evenly
    for num_shards in (1, 2, 5, 8, 37, 50):
        covered = [it for shard in range(num_shards) for it in _shard_items(items, shard, num_shards)]
        assert sorted(covered) == items
        assert len(covered) == len(items)


def test_shards_are_disjoint() -> None:
    """No work item appears in two shards."""
    items = list(range(20))
    seen: set[int] = set()
    for shard in range(4):
        part = _shard_items(items, shard, 4)
        assert not seen.intersection(part)
        seen.update(part)


def test_round_robin_balances_load() -> None:
    """Round-robin spreads items so shard sizes differ by at most one."""
    items = list(range(23))
    sizes = [len(_shard_items(items, s, 4)) for s in range(4)]
    assert max(sizes) - min(sizes) <= 1


def test_single_shard_returns_all() -> None:
    """A single shard is the whole work list."""
    items = list(range(10))
    assert _shard_items(items, 0, 1) == items


@pytest.mark.parametrize(
    "shard, num_shards",
    [(0, 0), (-1, 3), (3, 3), (5, 3)],
)
def test_invalid_shard_arguments_raise(shard: int, num_shards: int) -> None:
    """Out-of-range shard indices and non-positive counts are rejected."""
    with pytest.raises(ValueError):
        _shard_items([1, 2, 3], shard, num_shards)


def test_matched_cost_path_merged_vs_sharded() -> None:
    """A single shard writes the canonical name; multiple shards are suffixed."""
    merged = _matched_cost_path("steepness", 0, 1)
    assert merged.name == "matched_cost_steepness.json"
    sharded = _matched_cost_path("steepness", 2, 5)
    assert sharded.name == "matched_cost_steepness.shard2of5.json"


def test_sweep_specs_are_well_formed() -> None:
    """Each known sweep exposes a non-empty item list and positive discretization."""
    for which, spec in _MATCHED_COST_SWEEPS.items():
        assert spec.items, f"{which} has no work items"
        assert spec.modes > 0 and spec.nq > 0


def test_dfr_gets_more_budget_than_go() -> None:
    """The matched-cost multiplier gives plain DFR strictly more training."""
    assert _DFR_STEP_MULT > 1.0
