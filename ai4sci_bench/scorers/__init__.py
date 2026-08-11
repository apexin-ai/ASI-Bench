"""Built-in scorers for the benchmark framework."""

# Import all scorers to trigger registration
from ai4sci_bench.scorers.numerical import NumericalScorer
from ai4sci_bench.scorers.file_match import FileMatchScorer
from ai4sci_bench.scorers.code_analysis import CodeAnalysisScorer
from ai4sci_bench.scorers.composite import CompositeScorer
from ai4sci_bench.scorers.custom import load_custom_scorer
from ai4sci_bench.scorers.llm_judge import LLMJudgeScorer
from ai4sci_bench.scorers.multimodal import MultimodalScorer
from ai4sci_bench.scorers.agent_judge import AgentJudgeScorer
from ai4sci_bench.scorers._parse_utils import parse_judge_json

__all__ = [
    "NumericalScorer",
    "FileMatchScorer",
    "CodeAnalysisScorer",
    "CompositeScorer",
    "LLMJudgeScorer",
    "MultimodalScorer",
    "AgentJudgeScorer",
    "load_custom_scorer",
    "parse_judge_json",
]
