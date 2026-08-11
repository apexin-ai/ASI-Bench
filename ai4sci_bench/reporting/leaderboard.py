"""Leaderboard generation from multiple benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.reporting.results import RunReport

logger = get_logger(__name__)


@dataclass
class LeaderboardEntry:
    """A single entry (agent) on the leaderboard."""
    rank: int
    agent_name: str
    overall_score: float
    n_tasks: int
    n_instances: int
    scores_by_domain: dict[str, float] = field(default_factory=dict)
    scores_by_level: dict[str, float] = field(default_factory=dict)
    gates_pass_rate: float = 0.0
    total_execution_time: float = 0.0
    total_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Leaderboard:
    """Leaderboard comparing multiple agents."""
    entries: list[LeaderboardEntry] = field(default_factory=list)
    sort_by: str = "overall_score"

    def add_from_report(self, report: RunReport, metadata: dict[str, Any] | None = None) -> None:
        """Add an entry from a RunReport."""
        gates_pass_rate = 0.0
        if report.by_task:
            gates_pass_rate = sum(t.gates_pass_rate for t in report.by_task) / len(report.by_task)

        entry = LeaderboardEntry(
            rank=0,  # set after sorting
            agent_name=report.agent_name,
            overall_score=report.overall_mean_score,
            n_tasks=report.n_tasks,
            n_instances=report.n_instances,
            scores_by_domain={d.domain: d.mean_score for d in report.by_domain},
            scores_by_level=report.by_prompt_level,
            gates_pass_rate=gates_pass_rate,
            total_execution_time=report.total_execution_time,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        self._update_ranks()

    def _update_ranks(self) -> None:
        """Sort entries and assign ranks (higher score = lower rank number)."""
        self.entries.sort(key=lambda e: e.overall_score, reverse=True)
        for i, entry in enumerate(self.entries, 1):
            entry.rank = i

    def format_table(self, max_width: int = 100) -> str:
        """Format the leaderboard as a text table."""
        if not self.entries:
            return "No entries in leaderboard."

        lines = [
            "=" * max_width,
            "  ASI-Bench Leaderboard".center(max_width),
            "=" * max_width,
            "",
            f"  {'Rank':<6}{'Agent':<30}{'Score':<10}{'Tasks':<8}{'Gates%':<10}{'Time(s)':<10}",
            "  " + "-" * (max_width - 4),
        ]

        for entry in self.entries:
            lines.append(
                f"  {entry.rank:<6}{entry.agent_name:<30}"
                f"{entry.overall_score:<10.1f}{entry.n_tasks:<8}"
                f"{entry.gates_pass_rate * 100:<10.0f}{entry.total_execution_time:<10.0f}"
            )

        lines.append("")

        # Domain breakdown
        domains = set()
        for entry in self.entries:
            domains.update(entry.scores_by_domain.keys())

        if domains:
            lines.append("  By Domain:")
            header = f"  {'Agent':<25}" + "".join(f"{d:<15}" for d in sorted(domains))
            lines.append(header)
            lines.append("  " + "-" * (max_width - 4))
            for entry in self.entries:
                row = f"  {entry.agent_name:<25}"
                for d in sorted(domains):
                    score = entry.scores_by_domain.get(d, 0.0)
                    row += f"{score:<15.1f}"
                lines.append(row)
            lines.append("")

        # Prompt level breakdown
        levels = set()
        for entry in self.entries:
            levels.update(entry.scores_by_level.keys())

        if levels:
            lines.append("  By Prompt Level:")
            header = f"  {'Agent':<25}" + "".join(f"{l.upper():<15}" for l in sorted(levels))
            lines.append(header)
            lines.append("  " + "-" * (max_width - 4))
            for entry in self.entries:
                row = f"  {entry.agent_name:<25}"
                for l in sorted(levels):
                    score = entry.scores_by_level.get(l, 0.0)
                    row += f"{score:<15.1f}"
                lines.append(row)

        lines.append("=" * max_width)
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize leaderboard entries to dicts."""
        return [
            {
                "rank": e.rank,
                "agent_name": e.agent_name,
                "overall_score": e.overall_score,
                "n_tasks": e.n_tasks,
                "n_instances": e.n_instances,
                "scores_by_domain": e.scores_by_domain,
                "scores_by_level": e.scores_by_level,
                "gates_pass_rate": e.gates_pass_rate,
                "total_execution_time": e.total_execution_time,
                "total_cost": e.total_cost,
                "metadata": e.metadata,
            }
            for e in self.entries
        ]

    def save(self, path: Path) -> None:
        """Save leaderboard to JSON."""
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Leaderboard:
        """Load leaderboard from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        lb = cls()
        for d in data:
            lb.entries.append(LeaderboardEntry(
                rank=d["rank"],
                agent_name=d["agent_name"],
                overall_score=d["overall_score"],
                n_tasks=d["n_tasks"],
                n_instances=d["n_instances"],
                scores_by_domain=d.get("scores_by_domain", {}),
                scores_by_level=d.get("scores_by_level", {}),
                gates_pass_rate=d.get("gates_pass_rate", 0.0),
                total_execution_time=d.get("total_execution_time", 0.0),
                total_cost=d.get("total_cost", 0.0),
                metadata=d.get("metadata", {}),
            ))
        return lb
