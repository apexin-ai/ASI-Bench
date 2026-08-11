"""Review module: trajectory review and diagnosis using Claude Code as analysis engine.

Covers the full trajectory review requirements from the task review plan (round2/3):
1. Agent 是否正确理解任务目标
2. 过程与分数是否匹配（高分/低分都审）
3. B-level 分数梯度是否合理（跨 level 横向对比）
4. 异常行为检测（寻找 GT、硬编码、绕过核心计算等）
5. GT 泄露迹象检查
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai4sci_bench.core.logger import get_logger

logger = get_logger(__name__)

CAUSE_BUCKETS = ["Environment", "Task", "Agent"]

SCIENTIFIC_CATEGORIES = [
    "Prompt隐藏了与评分有关的信息",
    "Prompt信息模糊可能歧义",
    "Scorer与Prompt要求不一致",
    "Scorer存在漏洞(做错但高分)",
    "Agent物理边界条件理解错误",
    "Agent过于依赖标准模型",
    "Agent求解代码算法实现错误",
    "Agent取巧采用参数拟合/经典答案",
    "Agent尝试寻找GT/数据泄露",
    "环境/框架问题",
]


@dataclass
class Diagnosis:
    task_id: str
    prompt_level: str
    combo_label: str
    final_score: float
    what_implemented: str = ""
    main_mismatch: str = ""
    classification: str = ""
    cause_bucket: str = ""
    bottom_line: str = ""
    scorer_breakdown: str = ""
    evidence: str = ""
    task_understanding: str = ""
    process_score_match: str = ""
    anomaly_detected: str = ""
    raw_analysis: str = ""


@dataclass
class GradientAnalysis:
    """Cross-level score gradient analysis for a single task+combo."""
    task_id: str
    combo_label: str
    level_scores: dict[str, float] = field(default_factory=dict)
    is_monotonic: bool = True
    gradient_issues: str = ""
    b4_distractor_effective: str = ""
    raw_analysis: str = ""


@dataclass
class DiagnosisSummary:
    total_instances: int = 0
    reviewed_count: int = 0
    diagnoses: list[Diagnosis] = field(default_factory=list)
    gradient_analyses: list[GradientAnalysis] = field(default_factory=list)

    @property
    def bucket_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.diagnoses:
            bucket = d.cause_bucket or "Unknown"
            counts[bucket] = counts.get(bucket, 0) + 1
        return counts

    @property
    def category_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.diagnoses:
            cat = d.classification or "Unknown"
            counts[cat] = counts.get(cat, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Instance collection — no longer filtered by score threshold
# ---------------------------------------------------------------------------

def find_all_instances(
    output_dir: Path,
    combo_labels: list[str],
    prompt_levels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Find all result JSONs, optionally filtered by prompt levels."""
    results: list[dict[str, Any]] = []
    for label in combo_labels:
        combo_dir = output_dir / label
        if not combo_dir.exists():
            continue
        for json_file in sorted(combo_dir.rglob("*.json")):
            if json_file.name in (
                "run_metadata.json",
                "eval_config.json",
                "task_info.json",
                "framework_task_info.json",
            ):
                continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if "final_score" not in data or "task_id" not in data:
                continue
            level = data.get("prompt_level", "")
            if prompt_levels and level not in prompt_levels:
                continue
            data["_json_path"] = str(json_file)
            data["_combo_label"] = label
            results.append(data)
    return results


def find_low_score_instances(
    output_dir: Path,
    combo_labels: list[str],
    threshold: float = 10.0,
) -> list[dict[str, Any]]:
    """Find all result JSONs with final_score < threshold (backward compat)."""
    all_instances = find_all_instances(output_dir, combo_labels)
    return [d for d in all_instances if d["final_score"] < threshold]


# ---------------------------------------------------------------------------
# Material collection
# ---------------------------------------------------------------------------

def _collect_materials(
    result_data: dict[str, Any],
    tasks_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Collect all materials needed for diagnosis."""
    task_id = result_data["task_id"]
    domain, name = task_id.split(".", 1)
    task_dir = tasks_dir / domain / name

    materials: dict[str, str] = {}

    for fname in ("task.yaml", "custom_scorer.py", "reference_specs.md"):
        fpath = task_dir / fname
        if fpath.exists():
            materials[fname] = fpath.read_text(encoding="utf-8", errors="replace")

    prompt_level = result_data.get("prompt_level", "b2")
    prompt_file = task_dir / f"prompt_{prompt_level}.md"
    if prompt_file.exists():
        materials[f"prompt_{prompt_level}.md"] = prompt_file.read_text(
            encoding="utf-8", errors="replace"
        )

    gt_file = task_dir / "generate_gt.py"
    if gt_file.exists():
        materials["generate_gt.py"] = gt_file.read_text(
            encoding="utf-8", errors="replace"
        )

    score_info = {
        "final_score": result_data.get("final_score"),
        "gates_passed": result_data.get("gates_passed"),
        "gate_results": result_data.get("gate_results", []),
        "score_results": result_data.get("score_results", []),
        "status": result_data.get("status"),
    }
    materials["score_details.json"] = json.dumps(score_info, indent=2, ensure_ascii=False)

    result_path = Path(result_data["_json_path"])
    result_stem = result_path.stem
    result_dir = result_path.parent
    for suffix in (".agent_stdout.jsonl", ".agent_stderr.log", ".agent_model_output.md"):
        traj_file = result_dir / (result_stem + suffix)
        if traj_file.exists():
            content = traj_file.read_text(encoding="utf-8", errors="replace")
            if len(content) > 50000:
                content = content[:25000] + "\n\n... [TRUNCATED] ...\n\n" + content[-25000:]
            materials[f"trajectory{suffix}"] = content

    agent_output = result_data.get("agent_output", {})
    if isinstance(agent_output, dict):
        log = agent_output.get("log", "")
        if log:
            if len(log) > 30000:
                log = log[:15000] + "\n\n... [TRUNCATED] ...\n\n" + log[-15000:]
            materials["agent_log.txt"] = log

    return materials


# ---------------------------------------------------------------------------
# Per-instance trajectory review prompt (expanded for round2/3 requirements)
# ---------------------------------------------------------------------------

_REVIEW_PROMPT = """\
You are an expert benchmark reviewer for AI4Sci-Bench, a scientific computing \
benchmark that tests LLM agents on real research tasks across physics, chemistry, \
biology, math, and other domains.

**IMPORTANT: You are reviewing the TASK's correctness, NOT evaluating the agent's \
capability.** Problems you find in the agent's behavior often point to task design \
issues (unclear prompts, scorer bugs, GT leakage).

## Task: {task_id}
## Prompt Level: {prompt_level}
## Combo: {combo_label}
## Score: {final_score}

## Materials

{materials_block}

## Review Instructions

You must analyze the trajectory across ALL FIVE dimensions below. Every dimension \
requires a concrete judgment — do not skip any.

### Dimension 1: Agent 是否正确理解了任务目标
- Did the agent correctly understand the problem described in the prompt?
- If the agent misunderstood (e.g., treated 2D as 1D, used wrong physical model), \
this likely indicates the **Prompt is unclear** — record as a Task issue, not an Agent issue.
- If the agent's approach is reasonable but different from reference, check whether \
the scorer can handle alternative valid methods.

### Dimension 2: 过程与分数是否匹配
- **做对了但低分**: Agent's intermediate results look correct but final score is low \
→ scorer may be too strict, format validation too rigid, or gates misfiring.
- **做错了但高分**: Agent's approach is clearly wrong but score is not low \
→ scorer has a vulnerability, tolerance too wide, or key checks missing.
- **超时(status=timeout)导致0分**: Is the task's compute load too large or timeout too short?
- **崩溃(status=failed)非agent bug**: Sandbox environment issue or missing dependency?

### Dimension 3: 异常行为检测
Check for these specific anomalies (each may indicate a task design flaw):
- **Searching for GT**: Agent browses workspace for reference files, accesses `../`, \
reads `framework_task_info.json` → GT leakage risk.
- **Hardcoding values**: Agent writes "standard answer" numbers without computation → data contamination.
- **Bypassing core computation**: Agent uses redundant info in data/ to skip the intended \
calculation → data/ provides too much information.
- **Repeated failure on same step**: Agent retries the same operation multiple times → \
output format requirements unclear or sandbox missing dependencies.
- **Using unexpected external resources**: Web search for papers or reference code (when \
tools should be restricted).

### Dimension 4: Prompt-评分一致性
- Does the scorer evaluate ONLY what the prompt explicitly requires?
- Are there hidden rules in the scorer that the prompt doesn't mention?
- Does the scorer use overly strict format checks (regex on variable names, exact file names)?

### Dimension 5: GT 泄露迹象
- Does the prompt accidentally reveal expected numerical results or key intermediate values?
- Does data/ contain unnecessarily rich information that lets the agent bypass core computation?
- Does the agent's trajectory show signs of having accessed GT (checking parent dirs, etc.)?

## Required output format (use this EXACTLY, one line per field):

TASK_UNDERSTANDING: <did agent correctly understand the task? If not, what went wrong and why — is it a Prompt clarity issue?>
WHAT_IMPLEMENTED: <one sentence describing what the agent actually did>
PROCESS_SCORE_MATCH: <does the process match the score? Flag specific mismatches — "correct but low score" or "wrong but high score" with evidence>
MAIN_MISMATCH: <the key difference between agent's approach and the reference/scorer expectation>
ANOMALY_DETECTED: <list any anomalies from Dimension 3, or "None detected">
CLASSIFICATION: <one of: Prompt隐藏了与评分有关的信息 | Prompt信息模糊可能歧义 | Scorer与Prompt要求不一致 | Scorer存在漏洞(做错但高分) | Agent物理边界条件理解错误 | Agent过于依赖标准模型 | Agent求解代码算法实现错误 | Agent取巧采用参数拟合/经典答案 | Agent尝试寻找GT/数据泄露 | 环境/框架问题>
CAUSE_BUCKET: <Environment|Task|Agent>
SCORER_BREAKDOWN: <key scorer subcomponents and their scores>
EVIDENCE: <concrete evidence from trajectory/code/output supporting your judgment>
BOTTOM_LINE: <one sentence summary of the most important finding>
"""


# ---------------------------------------------------------------------------
# Cross-level gradient analysis prompt
# ---------------------------------------------------------------------------

_GRADIENT_PROMPT = """\
You are reviewing the B-level score gradient for a scientific benchmark task.

## Task: {task_id}
## Combo: {combo_label}

## Scores by level
{scores_block}

## All prompts for this task
{prompts_block}

## Instructions

Analyze the score gradient across B-levels. The expected design is:
- B1 has the most information → should score highest
- B2 has less → should score lower
- B3 has minimal information → should score lower still
- B4 has B2-level info PLUS distractor/misleading content → should score lower than B2

Answer these questions:

1. **Is the gradient monotonic?** B1 >= B2 >= B3 is expected (not strictly required). \
If B2 > B1 or B3 > B2, explain why — is it a prompt design issue or random variance?

2. **Is B4's distractor effective?** If B4 ≈ B3 or B4 > B3, the distractor content \
was likely ignored by the agent, meaning the B4 design may be ineffective.

3. **Are there anomalous inversions?** Any level scoring much higher than expected given \
its information content?

## Required output format:

IS_MONOTONIC: <yes/no>
GRADIENT_ISSUES: <describe specific gradient problems, or "None — gradient is reasonable">
B4_DISTRACTOR_EFFECTIVE: <yes/no/not_applicable — with brief explanation>
BOTTOM_LINE: <one sentence summary>
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_review_prompt(
    result_data: dict[str, Any],
    materials: dict[str, str],
) -> str:
    materials_block = ""
    for fname, content in materials.items():
        materials_block += f"\n### {fname}\n```\n{content}\n```\n"

    return _REVIEW_PROMPT.format(
        task_id=result_data.get("task_id", "unknown"),
        prompt_level=result_data.get("prompt_level", "unknown"),
        combo_label=result_data.get("_combo_label", "unknown"),
        final_score=result_data.get("final_score", "?"),
        materials_block=materials_block,
    )


def _build_gradient_prompt(
    task_id: str,
    combo_label: str,
    level_scores: dict[str, float],
    tasks_dir: Path,
) -> str:
    scores_block = "\n".join(
        f"- {level.upper()}: {score:.1f}" for level, score in sorted(level_scores.items())
    )

    domain, name = task_id.split(".", 1)
    task_dir = tasks_dir / domain / name
    prompts_block = ""
    for level in sorted(level_scores.keys()):
        pf = task_dir / f"prompt_{level}.md"
        if pf.exists():
            content = pf.read_text(encoding="utf-8", errors="replace")
            if len(content) > 8000:
                content = content[:4000] + "\n\n... [TRUNCATED] ...\n\n" + content[-4000:]
            prompts_block += f"\n### prompt_{level}.md\n```\n{content}\n```\n"

    return _GRADIENT_PROMPT.format(
        task_id=task_id,
        combo_label=combo_label,
        scores_block=scores_block,
        prompts_block=prompts_block,
    )


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

def _build_diagnosis_prompt(
    result_data: dict[str, Any],
    materials: dict[str, str],
) -> str:
    return _build_review_prompt(result_data, materials)


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_review_output(raw: str, result_data: dict[str, Any]) -> Diagnosis:
    d = Diagnosis(
        task_id=result_data.get("task_id", "unknown"),
        prompt_level=result_data.get("prompt_level", "unknown"),
        combo_label=result_data.get("_combo_label", "unknown"),
        final_score=result_data.get("final_score", 0.0),
        raw_analysis=raw,
    )

    field_map = {
        "TASK_UNDERSTANDING:": "task_understanding",
        "WHAT_IMPLEMENTED:": "what_implemented",
        "PROCESS_SCORE_MATCH:": "process_score_match",
        "MAIN_MISMATCH:": "main_mismatch",
        "ANOMALY_DETECTED:": "anomaly_detected",
        "CLASSIFICATION:": "classification",
        "CAUSE_BUCKET:": "cause_bucket",
        "SCORER_BREAKDOWN:": "scorer_breakdown",
        "EVIDENCE:": "evidence",
        "BOTTOM_LINE:": "bottom_line",
    }

    for line in raw.splitlines():
        line = line.strip()
        for prefix, attr in field_map.items():
            if line.startswith(prefix):
                setattr(d, attr, line.split(":", 1)[1].strip())
                break

    return d


def _parse_diagnosis_output(raw: str, result_data: dict[str, Any]) -> Diagnosis:
    return _parse_review_output(raw, result_data)


def _parse_gradient_output(raw: str, task_id: str, combo_label: str, level_scores: dict[str, float]) -> GradientAnalysis:
    g = GradientAnalysis(
        task_id=task_id,
        combo_label=combo_label,
        level_scores=dict(level_scores),
        raw_analysis=raw,
    )

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("IS_MONOTONIC:"):
            val = line.split(":", 1)[1].strip().lower()
            g.is_monotonic = val in ("yes", "true")
        elif line.startswith("GRADIENT_ISSUES:"):
            g.gradient_issues = line.split(":", 1)[1].strip()
        elif line.startswith("B4_DISTRACTOR_EFFECTIVE:"):
            g.b4_distractor_effective = line.split(":", 1)[1].strip()
        elif line.startswith("BOTTOM_LINE:"):
            g.gradient_issues = g.gradient_issues or line.split(":", 1)[1].strip()

    return g


# ---------------------------------------------------------------------------
# Claude Code invocation
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, timeout: int = 300) -> str:
    claude_path = shutil.which("claude")
    if not claude_path:
        logger.warning("Claude Code CLI not found")
        return "SKIPPED: Claude Code CLI not found"

    try:
        result = subprocess.run(
            [claude_path, "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        raw_output = result.stdout.strip()
        if not raw_output:
            raw_output = result.stderr.strip()
    except subprocess.TimeoutExpired:
        raw_output = "DIAGNOSIS TIMEOUT"
    except Exception as exc:
        raw_output = f"DIAGNOSIS ERROR: {exc}"

    return raw_output


# ---------------------------------------------------------------------------
# Single instance review
# ---------------------------------------------------------------------------

def run_diagnosis(
    result_data: dict[str, Any],
    tasks_dir: Path,
    output_dir: Path,
) -> Diagnosis:
    """Run Claude Code review on a single instance (any score, not just low)."""
    materials = _collect_materials(result_data, tasks_dir, output_dir)
    prompt = _build_review_prompt(result_data, materials)

    raw_output = _call_claude(prompt)
    if raw_output.startswith("SKIPPED:"):
        return Diagnosis(
            task_id=result_data.get("task_id", "unknown"),
            prompt_level=result_data.get("prompt_level", "unknown"),
            combo_label=result_data.get("_combo_label", "unknown"),
            final_score=result_data.get("final_score", 0.0),
            bottom_line=raw_output,
        )

    return _parse_review_output(raw_output, result_data)


# ---------------------------------------------------------------------------
# Gradient analysis
# ---------------------------------------------------------------------------

def run_gradient_analysis(
    task_id: str,
    combo_label: str,
    level_scores: dict[str, float],
    tasks_dir: Path,
) -> GradientAnalysis:
    """Analyze score gradient across B-levels for one task+combo."""
    if len(level_scores) < 2:
        return GradientAnalysis(
            task_id=task_id,
            combo_label=combo_label,
            level_scores=dict(level_scores),
            gradient_issues="Only one level — gradient analysis not applicable",
        )

    prompt = _build_gradient_prompt(task_id, combo_label, level_scores, tasks_dir)
    raw_output = _call_claude(prompt, timeout=120)

    if raw_output.startswith("SKIPPED:") or raw_output.startswith("DIAGNOSIS"):
        return GradientAnalysis(
            task_id=task_id,
            combo_label=combo_label,
            level_scores=dict(level_scores),
            gradient_issues=raw_output,
        )

    return _parse_gradient_output(raw_output, task_id, combo_label, level_scores)


# ---------------------------------------------------------------------------
# Full review pipeline
# ---------------------------------------------------------------------------

def run_all_diagnoses(
    output_dir: Path,
    combo_labels: list[str],
    tasks_dir: Path,
    threshold: float = 10.0,
    total_instances: int = 0,
    prompt_levels: list[str] | None = None,
    review_all: bool = False,
) -> DiagnosisSummary:
    """Run trajectory review on instances.

    When review_all=True, reviews ALL instances (not just low-score) and also
    performs cross-level gradient analysis — matching the round2/3 trajectory
    review requirements.

    When review_all=False, falls back to the original low-score-only behavior.

    prompt_levels controls which B-levels are reviewed. If None, all levels found
    in results are reviewed.
    """
    if review_all:
        all_instances = find_all_instances(output_dir, combo_labels, prompt_levels)
    else:
        all_instances = find_low_score_instances(output_dir, combo_labels, threshold)
        if prompt_levels:
            all_instances = [
                d for d in all_instances
                if d.get("prompt_level", "") in prompt_levels
            ]

    summary = DiagnosisSummary(
        total_instances=total_instances or len(all_instances),
        reviewed_count=len(all_instances),
    )

    # Phase 1: Per-instance trajectory review
    for i, result_data in enumerate(all_instances, 1):
        tid = result_data.get("task_id", "?")
        level = result_data.get("prompt_level", "?")
        label = result_data.get("_combo_label", "?")
        score = result_data.get("final_score", 0)
        logger.info(
            "Reviewing [%d/%d]: %s %s — %s (score: %.1f)",
            i, len(all_instances), tid, level, label, score,
        )
        diagnosis = run_diagnosis(result_data, tasks_dir, output_dir)
        summary.diagnoses.append(diagnosis)

    # Phase 2: Cross-level gradient analysis (only when reviewing all)
    if review_all:
        task_combo_scores: dict[tuple[str, str], dict[str, float]] = {}
        for d in all_instances:
            key = (d["task_id"], d["_combo_label"])
            level = d.get("prompt_level", "")
            if level:
                task_combo_scores.setdefault(key, {})[level] = d["final_score"]

        for (tid, label), level_scores in task_combo_scores.items():
            if len(level_scores) >= 2:
                logger.info("Gradient analysis: %s — %s", tid, label)
                ga = run_gradient_analysis(tid, label, level_scores, tasks_dir)
                summary.gradient_analyses.append(ga)

    return summary


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_diagnosis_report(
    summary: DiagnosisSummary,
    output_dir: Path,
) -> None:
    """Write review reports: per-instance files + gradient reports + summary.md."""
    diag_dir = output_dir / "diagnosis"
    diag_dir.mkdir(parents=True, exist_ok=True)

    for d in summary.diagnoses:
        fname = f"{d.task_id}__{d.prompt_level}__{d.combo_label}.md"
        fname = fname.replace("/", "_").replace(" ", "_")
        report = _format_single_diagnosis(d)
        (diag_dir / fname).write_text(report, encoding="utf-8")

    for g in summary.gradient_analyses:
        fname = f"gradient__{g.task_id}__{g.combo_label}.md"
        fname = fname.replace("/", "_").replace(" ", "_")
        report = _format_gradient_report(g)
        (diag_dir / fname).write_text(report, encoding="utf-8")

    summary_text = _format_summary(summary)
    (diag_dir / "summary.md").write_text(summary_text, encoding="utf-8")


def _format_single_diagnosis(d: Diagnosis) -> str:
    lines = [
        f"# {d.task_id} {d.prompt_level.upper()} — {d.combo_label}\n",
        "## Score",
        f"- final_score: {d.final_score}",
        f"- scorer breakdown: {d.scorer_breakdown}\n",
        "## 任务理解",
        f"- {d.task_understanding}\n",
        "## Agent 实现",
        f"- {d.what_implemented}\n",
        "## 过程-分数匹配",
        f"- {d.process_score_match}\n",
        "## 与参考实现的主要差异",
        f"- {d.main_mismatch}\n",
        "## 异常行为",
        f"- {d.anomaly_detected}\n",
        "## 分类",
        f"- 科学因果: {d.classification}",
        f"- 归因桶: {d.cause_bucket}\n",
        "## 证据",
        f"- {d.evidence}\n",
        "## 结论",
        f"- {d.bottom_line}\n",
    ]
    return "\n".join(lines)


def _format_gradient_report(g: GradientAnalysis) -> str:
    lines = [
        f"# 分数梯度分析: {g.task_id} — {g.combo_label}\n",
        "## 各 Level 分数",
    ]
    for level, score in sorted(g.level_scores.items()):
        lines.append(f"- {level.upper()}: {score:.1f}")
    lines.extend([
        f"\n## 单调性: {'是' if g.is_monotonic else '否'}",
        f"\n## 梯度问题",
        f"- {g.gradient_issues}",
        f"\n## B4 干扰有效性",
        f"- {g.b4_distractor_effective}",
        "",
    ])
    return "\n".join(lines)


def _format_summary(summary: DiagnosisSummary) -> str:
    lines = [
        "# Trajectory 审核汇总\n",
        "## 统计",
        f"- 总评测实例: {summary.total_instances}",
        f"- 已审核实例: {summary.reviewed_count}",
        f"- 诊断完成: {len(summary.diagnoses)}",
        f"- 梯度分析: {len(summary.gradient_analyses)}\n",
    ]

    # Bucket distribution
    lines.extend([
        "## 归因分布\n",
        "| 归因桶 | 数量 | 占比 |",
        "|--------|------|------|",
    ])
    bucket_dist = summary.bucket_distribution
    total = max(len(summary.diagnoses), 1)
    for bucket, count in sorted(bucket_dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        lines.append(f"| {bucket} | {count} | {pct:.0f}% |")

    # Category distribution
    lines.extend([
        "\n## 科学因果分布\n",
        "| 分类 | 数量 |",
        "|------|------|",
    ])
    cat_dist = summary.category_distribution
    for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {count} |")

    # Gradient analysis summary
    if summary.gradient_analyses:
        lines.extend([
            "\n## 分数梯度分析\n",
            "| Task | Combo | 分数 | 单调 | 梯度问题 | B4干扰 |",
            "|------|-------|------|------|---------|--------|",
        ])
        for g in summary.gradient_analyses:
            scores_str = " / ".join(
                f"{l.upper()}={s:.1f}" for l, s in sorted(g.level_scores.items())
            )
            mono = "✓" if g.is_monotonic else "✗"
            issue = g.gradient_issues[:60] if g.gradient_issues else "—"
            b4 = g.b4_distractor_effective[:30] if g.b4_distractor_effective else "—"
            lines.append(f"| {g.task_id} | {g.combo_label} | {scores_str} | {mono} | {issue} | {b4} |")

    # Task issues
    task_issues = [d for d in summary.diagnoses if d.cause_bucket == "Task"]
    if task_issues:
        lines.extend([
            "\n## 需关注的 Task 问题\n",
            "以下实例被归因为 Task 设计问题：\n",
        ])
        for d in task_issues:
            lines.append(f"- `{d.task_id}` ({d.prompt_level}, score={d.final_score:.1f}): {d.bottom_line}")

    # Scorer issues (high score but wrong process)
    scorer_issues = [d for d in summary.diagnoses if "Scorer" in (d.classification or "")]
    if scorer_issues:
        lines.extend([
            "\n## Scorer 漏洞\n",
            "以下实例可能存在 Scorer 问题（做错但高分，或评分与 Prompt 不一致）：\n",
        ])
        for d in scorer_issues:
            lines.append(f"- `{d.task_id}` ({d.prompt_level}, score={d.final_score:.1f}): {d.bottom_line}")

    # Per-instance index
    lines.extend([
        "\n## 逐实例索引\n",
        "| Task | Level | Combo | Score | 归因 | 分类 | 过程-分数匹配 |",
        "|------|-------|-------|-------|------|------|-------------|",
    ])
    for d in summary.diagnoses:
        match_short = (d.process_score_match or "—")[:50]
        lines.append(
            f"| {d.task_id} | {d.prompt_level} | {d.combo_label} "
            f"| {d.final_score:.1f} | {d.cause_bucket} | {d.classification} | {match_short} |"
        )

    return "\n".join(lines)
