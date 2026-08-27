"""Built-in agent adapters."""

from ai4sci_bench.adapters.subprocess_base import SubprocessAgentAdapter
from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter
from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from ai4sci_bench.adapters.cli_agent import CLIAgentAdapter
from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
from ai4sci_bench.adapters.openhands_agent import OpenHandsAdapter
from ai4sci_bench.adapters.hermes_agent import HermesAgentAdapter
from ai4sci_bench.adapters.codewhale_agent import CodeWhaleAdapter
from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter
from ai4sci_bench.adapters.antigravity_cli import AntigravityCLIAdapter
from ai4sci_bench.adapters.mimo_code_cli import MiMoCodeCLIAdapter
from ai4sci_bench.adapters.pi_cli import PiCLIAdapter
from ai4sci_bench.adapters.opencode_cli import OpenCodeCLIAdapter

__all__ = [
    "SubprocessAgentAdapter",
    "DirectLLMAdapter",
    "ClaudeCodeCLIAdapter",
    "CLIAgentAdapter",
    "CodexCLIAdapter",
    "OpenHandsAdapter",
    "HermesAgentAdapter",
    "CodeWhaleAdapter",
    "KimiCodeCLIAdapter",
    "AntigravityCLIAdapter",
    "MiMoCodeCLIAdapter",
    "PiCLIAdapter",
    "OpenCodeCLIAdapter",
]
