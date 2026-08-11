"""Runner — orchestration, execution, and sandbox management."""

__all__ = ["BenchmarkOrchestrator", "RunConfig"]


def __getattr__(name: str):
    if name in __all__:
        from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig

        exports = {
            "BenchmarkOrchestrator": BenchmarkOrchestrator,
            "RunConfig": RunConfig,
        }
        return exports[name]
    raise AttributeError(name)
