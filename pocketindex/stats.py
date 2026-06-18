"""Processing statistics for the PocketIndex engine.

Mirrors the observability model of upstream CocoIndex (``UpdateStats`` /
``ComponentStats``) so that ``Target = F(Source)`` runs report how many source
items were added, reprocessed, left unchanged, deleted, or errored. This is the
backbone of monitoring and log cross-checking for incremental pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ComponentStats:
    """Per-component (per-processor) processing counters.

    Mirrors upstream ``ComponentStats`` semantics so monitoring output is
    comparable across the two engines.
    """

    num_execution_starts: int = 0
    num_unchanged: int = 0
    num_adds: int = 0
    num_deletes: int = 0
    num_reprocesses: int = 0
    num_errors: int = 0

    @property
    def num_processed(self) -> int:
        return self.num_unchanged + self.num_adds + self.num_deletes + self.num_reprocesses

    @property
    def num_finished(self) -> int:
        return self.num_processed + self.num_errors

    @property
    def num_in_progress(self) -> int:
        return max(0, self.num_execution_starts - self.num_finished)

    def merge(self, other: "ComponentStats") -> None:
        self.num_execution_starts += other.num_execution_starts
        self.num_unchanged += other.num_unchanged
        self.num_adds += other.num_adds
        self.num_deletes += other.num_deletes
        self.num_reprocesses += other.num_reprocesses
        self.num_errors += other.num_errors

    def __str__(self) -> str:
        return (
            f"adds={self.num_adds} reprocesses={self.num_reprocesses} "
            f"unchanged={self.num_unchanged} deletes={self.num_deletes} "
            f"errors={self.num_errors} in_progress={self.num_in_progress}"
        )


@dataclass
class UpdateStats:
    """Aggregate snapshot of a pipeline run, keyed by component name.

    Mirrors upstream ``UpdateStats``: ``by_component`` holds per-processor
    counters and ``total`` aggregates across all of them.
    """

    by_component: Dict[str, ComponentStats] = field(default_factory=dict)

    def component(self, name: str) -> ComponentStats:
        """Return (creating if needed) the counters for a named component."""
        stats = self.by_component.get(name)
        if stats is None:
            stats = ComponentStats()
            self.by_component[name] = stats
        return stats

    @property
    def total(self) -> ComponentStats:
        agg = ComponentStats()
        for stats in self.by_component.values():
            agg.merge(stats)
        return agg

    def __str__(self) -> str:
        lines = []
        for name, stats in self.by_component.items():
            lines.append(f"  {name}: {stats}")
        lines.append(f"  total: {self.total}")
        return "\n".join(lines)
