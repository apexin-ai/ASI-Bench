"""Core data models and interfaces."""

from ai4sci_bench.core.types import (
    AgentOutput,
    EvalResult,
    PromptLevel,
    RunStatus,
    ScoreDetail,
    TaskInstance,
    TaskLifecycle,
)
from ai4sci_bench.core.scorer import Scorer, get_scorer, register_scorer
from ai4sci_bench.core.agent_interface import AgentAdapter
from ai4sci_bench.core.task import TaskLoader

__all__ = [
    "AgentAdapter",
    "AgentOutput",
    "EvalResult",
    "PromptLevel",
    "RunStatus",
    "ScoreDetail",
    "Scorer",
    "TaskInstance",
    "TaskLifecycle",
    "TaskLoader",
    "get_scorer",
    "register_scorer",
]
