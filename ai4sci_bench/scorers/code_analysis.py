"""Code analysis scorer — static analysis of agent-generated code."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


@register_scorer("code_analysis")
class CodeAnalysisScorer(Scorer):
    """Static analysis of agent-generated code.

    Checks:
      - required patterns must be present
      - forbidden patterns must be absent
      - minimum pattern occurrence count
      - line count range
    """

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = config.get("weight", 1.0)
        target_file = config.get("target_file", "simulation.py")
        checks = config.get("checks", [])
        scoring_mode = config.get("scoring_mode", "all_or_nothing")

        file_path = pred_dir / target_file
        if not file_path.exists():
            return ScoreDetail(
                scorer_name="code_analysis",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"error": f"File not found: {target_file}"},
                message=f"Target file not found: {target_file}",
            )

        code = file_path.read_text(encoding="utf-8")
        check_results = []
        all_passed = True

        for check in checks:
            result = self._run_check(code, check, pred_dir=pred_dir)
            check_results.append(result)
            if not result["passed"]:
                all_passed = False

        # Line count check
        if "min_lines" in config or "max_lines" in config:
            lines = len(code.splitlines())
            line_check = {"check": "line_count", "lines": lines, "passed": True}
            if "min_lines" in config and lines < config["min_lines"]:
                line_check["passed"] = False
                line_check["message"] = f"Too few lines: {lines} < {config['min_lines']}"
            if "max_lines" in config and lines > config["max_lines"]:
                line_check["passed"] = False
                line_check["message"] = f"Too many lines: {lines} > {config['max_lines']}"
            check_results.append(line_check)
            if not line_check["passed"]:
                all_passed = False

        details = {"checks": check_results}
        # Count kernel occurrences if applicable
        for check in checks:
            pattern = check.get("pattern", "")
            if "@ti.kernel" in pattern or "kernel" in pattern.lower():
                count = len(re.findall(pattern, code))
                details["kernel_count"] = count

        failed_checks = [r.get("message") for r in check_results if not r["passed"] and r.get("message")]
        message = "; ".join(failed_checks) if failed_checks else ""
        if not message and not all_passed:
            message = "Some code analysis checks failed"

        score = self._compute_score(check_results, weight, scoring_mode)
        return ScoreDetail(
            scorer_name="code_analysis",
            score=score,
            max_score=weight,
            passed=all_passed,
            details={**details, "scoring_mode": scoring_mode},
            message=message,
        )

    def _compute_score(self, check_results: list[dict], weight: float, scoring_mode: str) -> float:
        if scoring_mode == "all_or_nothing":
            return weight if all(result.get("passed", False) for result in check_results) else 0.0
        if scoring_mode == "per_check":
            if not check_results:
                return weight
            passed = sum(1 for result in check_results if result.get("passed", False))
            return weight * passed / len(check_results)
        raise ValueError(f"Unknown code_analysis scoring_mode: {scoring_mode}")

    def _run_check(self, code: str, check: dict, pred_dir: Path | None = None) -> dict:
        if "forbidden_imports" in check:
            return self._run_forbidden_imports_check(code, check)

        pattern = check.get("pattern", "")
        required = check.get("required", False)
        forbidden = check.get("forbidden", False)
        min_count = check.get("min_count")

        matches = re.findall(pattern, code)
        count = len(matches)

        result = {"pattern": pattern, "count": count, "passed": True}

        if required and count == 0:
            satisfied_file = self._matching_output_file(pred_dir, pattern)
            if satisfied_file:
                result["satisfied_by_file"] = satisfied_file
            else:
                result["passed"] = False
                result["message"] = f"Required pattern not found: {pattern}"
        elif forbidden and count > 0:
            clean_code = self._strip_comments_and_docstrings(code)
            clean_count = len(re.findall(pattern, clean_code))
            if clean_count > 0:
                result["passed"] = False
                result["message"] = f"Forbidden pattern found: {pattern} ({clean_count} occurrences in code)"
            else:
                result["message"] = (
                    f"Forbidden pattern found only in comments/docstrings "
                    f"({count} occurrences), ignored"
                )
        elif min_count is not None and count < min_count:
            result["passed"] = False
            result["message"] = f"Pattern '{pattern}' found {count} times, need >= {min_count}"

        return result

    def _run_forbidden_imports_check(self, code: str, check: dict) -> dict:
        """Reject real Python imports while ignoring comments and docstrings."""
        forbidden_imports = [
            str(module).strip()
            for module in check.get("forbidden_imports", [])
            if str(module).strip()
        ]
        result = {
            "check": "forbidden_imports",
            "forbidden_imports": forbidden_imports,
            "matched_imports": [],
            "count": 0,
            "passed": True,
        }
        if not forbidden_imports:
            return result

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            result["passed"] = False
            result["message"] = (
                f"Could not parse Python for forbidden_imports: {exc.msg}"
            )
            result["syntax_error"] = {
                "lineno": exc.lineno,
                "offset": exc.offset,
            }
            return result

        imported_modules = self._collect_imported_modules(tree)
        matched = sorted(
            {
                imported
                for imported in imported_modules
                for forbidden in forbidden_imports
                if self._module_matches_forbidden(imported, forbidden)
            }
        )
        if matched:
            result["matched_imports"] = matched
            result["count"] = len(matched)
            result["passed"] = False
            result["message"] = f"Forbidden import found: {', '.join(matched)}"

        return result

    @staticmethod
    def _collect_imported_modules(tree: ast.AST) -> list[str]:
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                base = node.module or ""
                if base:
                    imported.append(base)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imported.append(f"{base}.{alias.name}" if base else alias.name)
        return imported

    @staticmethod
    def _module_matches_forbidden(imported: str, forbidden: str) -> bool:
        return imported == forbidden or imported.startswith(f"{forbidden}.")

    @staticmethod
    def _strip_comments_and_docstrings(code: str) -> str:
        """Return *code* with comments and docstrings removed (line count preserved)."""
        lines = code.splitlines()
        stripped: list[str] = []
        for line in lines:
            in_string = False
            quote_char: str | None = None
            result_chars: list[str] = []
            i = 0
            while i < len(line):
                ch = line[i]
                if in_string:
                    result_chars.append(ch)
                    if ch == '\\':
                        i += 1
                        if i < len(line):
                            result_chars.append(line[i])
                    elif ch == quote_char:
                        in_string = False
                elif ch in ('"', "'"):
                    in_string = True
                    quote_char = ch
                    result_chars.append(ch)
                elif ch == '#':
                    break
                else:
                    result_chars.append(ch)
                i += 1
            stripped.append(''.join(result_chars))

        code_no_comments = '\n'.join(stripped)

        try:
            tree = ast.parse(code_no_comments)
        except SyntaxError:
            return code_no_comments

        docstring_lines: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                for ln in range(node.lineno, node.end_lineno + 1):
                    docstring_lines.add(ln)

        final_lines: list[str] = []
        for i, line in enumerate(stripped, 1):
            final_lines.append('' if i in docstring_lines else line)

        return '\n'.join(final_lines)

    def _matching_output_file(self, pred_dir: Path | None, pattern: str) -> str | None:
        """Treat filename-style required patterns as satisfied by produced files.

        This keeps code_analysis useful for static implementation checks while
        avoiding a hard failure when a task separately verifies that an artifact
        such as overview.png was actually produced.
        """
        if pred_dir is None:
            return None

        candidates = self._literal_file_candidates(pattern)
        if not candidates:
            return None

        for candidate in candidates:
            direct = pred_dir / candidate
            if direct.is_file():
                return direct.relative_to(pred_dir).as_posix()

            basename = Path(candidate).name
            for path in pred_dir.rglob(basename):
                if path.is_file():
                    return path.relative_to(pred_dir).as_posix()

        return None

    @staticmethod
    def _literal_file_candidates(pattern: str) -> list[str]:
        matches = re.findall(
            r"([A-Za-z0-9_./-]+)\\\.(png|jpe?g|csv|npy|json|txt|pdf|svg)",
            pattern,
        )
        return [f"{stem}.{suffix}" for stem, suffix in matches]
