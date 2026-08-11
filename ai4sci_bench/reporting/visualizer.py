"""Result visualization — tables and charts for benchmark results.

Generates text-based tables (always available) and optional matplotlib charts
(when matplotlib is installed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.reporting.results import RunReport

logger = get_logger(__name__)


class ResultVisualizer:
    """Generate visual reports from benchmark results."""

    def __init__(self, report: RunReport):
        self.report = report

    # ── Text Tables ──────────────────────────────────────────────────────

    def task_table(self) -> str:
        """Format per-task results as a text table."""
        if not self.report.by_task:
            return "No task results available."

        header = (
            f"  {'Task ID':<35}{'N':<5}{'Mean':<8}{'Min':<8}{'Max':<8}"
            f"{'Std':<8}{'Gates%':<8}"
        )
        sep = "  " + "-" * 78

        lines = [sep, header, sep]
        for ts in sorted(self.report.by_task, key=lambda t: t.task_id):
            lines.append(
                f"  {ts.task_id:<35}{ts.n_instances:<5}"
                f"{ts.mean_score:<8.1f}{ts.min_score:<8.1f}{ts.max_score:<8.1f}"
                f"{ts.std_score:<8.1f}{ts.gates_pass_rate * 100:<8.0f}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def domain_table(self) -> str:
        """Format per-domain results as a text table."""
        if not self.report.by_domain:
            return "No domain results available."

        header = f"  {'Domain':<20}{'Tasks':<8}{'Mean Score':<12}"
        sep = "  " + "-" * 38

        lines = [sep, header, sep]
        for ds in sorted(self.report.by_domain, key=lambda d: d.domain):
            lines.append(
                f"  {ds.domain:<20}{ds.n_tasks:<8}{ds.mean_score:<12.1f}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def prompt_level_table(self) -> str:
        """Format per-prompt-level results as a text table."""
        if not self.report.by_prompt_level:
            return "No prompt level results available."

        header = f"  {'Level':<10}{'Mean Score':<12}"
        sep = "  " + "-" * 20

        lines = [sep, header, sep]
        for level, score in sorted(self.report.by_prompt_level.items()):
            lines.append(f"  {level.upper():<10}{score:<12.1f}")
        lines.append(sep)
        return "\n".join(lines)

    def error_distribution_table(self) -> str:
        """Format error distribution as a text table."""
        if not self.report.error_distribution:
            return "No error analysis data available."

        total = sum(self.report.error_distribution.values())
        header = f"  {'Category':<25}{'Count':<8}{'Percentage':<12}"
        sep = "  " + "-" * 43

        lines = [sep, header, sep]
        for cat, count in sorted(
            self.report.error_distribution.items(), key=lambda x: -x[1]
        ):
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  {cat:<25}{count:<8}{pct:<12.1f}%")
        lines.append(sep)
        lines.append(f"  {'Total':<25}{total:<8}")
        lines.append(sep)
        return "\n".join(lines)

    def full_report(self) -> str:
        """Generate a comprehensive text report combining all tables."""
        sections = [
            self.report.format_summary(),
            "",
            "Per-Task Breakdown:",
            self.task_table(),
            "",
            "Per-Domain Breakdown:",
            self.domain_table(),
            "",
            "Per-Prompt-Level Breakdown:",
            self.prompt_level_table(),
        ]
        if self.report.error_distribution:
            sections += [
                "",
                "Error Distribution:",
                self.error_distribution_table(),
            ]
        return "\n".join(sections)

    # ── Chart Generation (optional, requires matplotlib) ─────────────────

    def save_charts(self, output_dir: Path) -> list[Path]:
        """Generate and save charts as PNG files.

        Returns list of saved file paths. Requires matplotlib.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed. Skipping chart generation.")
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        # 1. Domain scores bar chart
        if self.report.by_domain:
            fig, ax = plt.subplots(figsize=(10, 6))
            domains = [d.domain for d in self.report.by_domain]
            scores = [d.mean_score for d in self.report.by_domain]
            ax.barh(domains, scores, color="#4a90d9")
            ax.set_xlabel("Mean Score")
            ax.set_title(f"Scores by Domain — {self.report.agent_name}")
            ax.set_xlim(0, 100)
            ax.invert_yaxis()
            fig.tight_layout()
            path = output_dir / "domain_scores.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

        # 2. Prompt level scores bar chart
        if self.report.by_prompt_level:
            fig, ax = plt.subplots(figsize=(8, 5))
            levels = sorted(self.report.by_prompt_level.keys())
            scores = [self.report.by_prompt_level[l] for l in levels]
            ax.bar([l.upper() for l in levels], scores, color=["#27ae60", "#f39c12", "#e74c3c"])
            ax.set_ylabel("Mean Score")
            ax.set_title(f"Scores by Prompt Level — {self.report.agent_name}")
            ax.set_ylim(0, 100)
            fig.tight_layout()
            path = output_dir / "prompt_level_scores.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

        # 3. Error distribution pie chart
        if self.report.error_distribution:
            fig, ax = plt.subplots(figsize=(8, 8))
            labels = list(self.report.error_distribution.keys())
            sizes = list(self.report.error_distribution.values())
            ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=140)
            ax.set_title(f"Error Distribution — {self.report.agent_name}")
            fig.tight_layout()
            path = output_dir / "error_distribution.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

        # 4. Per-task score distribution
        if self.report.by_task:
            fig, ax = plt.subplots(figsize=(12, max(6, len(self.report.by_task) * 0.4)))
            tasks = sorted(self.report.by_task, key=lambda t: t.mean_score)
            names = [t.task_id for t in tasks]
            means = [t.mean_score for t in tasks]
            stds = [t.std_score for t in tasks]
            ax.barh(names, means, xerr=stds, color="#4a90d9", capsize=3)
            ax.set_xlabel("Score")
            ax.set_title(f"Per-Task Scores — {self.report.agent_name}")
            ax.set_xlim(0, 100)
            fig.tight_layout()
            path = output_dir / "task_scores.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

        return saved

    def save_json(self, path: Path) -> None:
        """Save the full report data as JSON for downstream consumption."""
        data = {
            "agent_name": self.report.agent_name,
            "n_tasks": self.report.n_tasks,
            "n_instances": self.report.n_instances,
            "overall_mean_score": self.report.overall_mean_score,
            "by_task": [
                {
                    "task_id": ts.task_id,
                    "n_instances": ts.n_instances,
                    "mean_score": ts.mean_score,
                    "min_score": ts.min_score,
                    "max_score": ts.max_score,
                    "std_score": ts.std_score,
                    "gates_pass_rate": ts.gates_pass_rate,
                    "scores_by_level": ts.scores_by_level,
                }
                for ts in self.report.by_task
            ],
            "by_domain": [
                {
                    "domain": ds.domain,
                    "n_tasks": ds.n_tasks,
                    "mean_score": ds.mean_score,
                }
                for ds in self.report.by_domain
            ],
            "by_prompt_level": self.report.by_prompt_level,
            "error_distribution": self.report.error_distribution,
            "total_execution_time": self.report.total_execution_time,
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
