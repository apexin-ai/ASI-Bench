"""Parameter space definition and sampling."""

from __future__ import annotations

import random
from typing import Any


class ParamSpace:
    """Parameter space definition and sampling."""

    def __init__(self, param_definitions: dict[str, dict]):
        """Initialize from task.yaml generation.parameters.

        Args:
            param_definitions: e.g. {"grid_size": {"type": "int", "range": [127, 511], "default": 255}}
        """
        self.param_definitions = param_definitions

    @property
    def defaults(self) -> dict[str, Any]:
        """Return default parameter values."""
        return {
            name: spec.get("default")
            for name, spec in self.param_definitions.items()
            if "default" in spec
        }

    def sample(
        self,
        n: int,
        strategy: str = "random",
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Sample n parameter sets from the space.

        Strategies:
          random             uniform random sampling
          latin_hypercube    LHS (better space coverage)
          grid               full grid (combinatorial)
          boundary           boundary/edge values
        """
        if strategy == "random":
            return [self._sample_random(seed, i) for i in range(n)]
        elif strategy == "latin_hypercube":
            return self._sample_lhs(n, seed)
        elif strategy == "grid":
            return self._sample_grid(n)
        elif strategy == "boundary":
            return self._sample_boundary(n)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def _sample_random(self, seed: int | None, idx: int) -> dict[str, Any]:
        """Sample one parameter set using seed+idx."""
        return _sample_params(seed or 0, idx, self.param_definitions)

    def _sample_lhs(self, n: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Latin Hypercube Sampling for better coverage."""
        rng = random.Random(seed)
        params_list = []

        # For each parameter, create n evenly-spaced bins and shuffle
        bins = {}
        for name, spec in self.param_definitions.items():
            if spec["type"] == "choice":
                options = spec["options"]
                bin_values = [options[i % len(options)] for i in range(n)]
                rng.shuffle(bin_values)
                bins[name] = bin_values
            else:
                lo, hi = spec["range"]
                bin_values = []
                for i in range(n):
                    low_bound = lo + (hi - lo) * i / n
                    high_bound = lo + (hi - lo) * (i + 1) / n
                    if spec["type"] == "int":
                        val = rng.randint(int(low_bound), int(high_bound))
                    else:
                        val = rng.uniform(low_bound, high_bound)
                    bin_values.append(val)
                rng.shuffle(bin_values)
                bins[name] = bin_values

        for i in range(n):
            params = {name: bins[name][i] for name in self.param_definitions}
            params["seed"] = random.Random((seed or 0) + i + n).randint(0, 2**31 - 1)
            params_list.append(params)

        return params_list

    def _sample_grid(self, n: int) -> list[dict[str, Any]]:
        """Grid sampling (returns up to n samples from combinatorial grid)."""
        import itertools

        grid_values = {}
        for name, spec in self.param_definitions.items():
            if spec["type"] == "choice":
                grid_values[name] = spec["options"]
            elif spec["type"] == "int":
                lo, hi = spec["range"]
                step = max(1, (hi - lo) // max(1, int(n ** (1 / len(self.param_definitions)))))
                grid_values[name] = list(range(lo, hi + 1, step))
            else:
                lo, hi = spec["range"]
                steps = max(2, int(n ** (1 / len(self.param_definitions))))
                grid_values[name] = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]

        combos = list(itertools.product(*grid_values.values()))[:n]
        names = list(grid_values.keys())
        return [dict(zip(names, combo)) for combo in combos]

    def _sample_boundary(self, n: int) -> list[dict[str, Any]]:
        """Sample boundary/edge values for stress testing."""
        samples = []
        for name, spec in self.param_definitions.items():
            if spec["type"] in ("int", "float"):
                lo, hi = spec["range"]
                # Min values
                params = {**self.defaults, name: lo}
                samples.append(params)
                # Max values
                params = {**self.defaults, name: hi}
                samples.append(params)
        return samples[:n]


def _sample_params(sampling_seed: int, idx: int, param_space: dict) -> dict[str, Any]:
    """Deterministically sample one parameter set from (sampling_seed, idx).

    Used by the on-the-fly mode.
    """
    rng = random.Random(sampling_seed + idx)
    params = {}
    for name, spec in param_space.items():
        if spec["type"] == "int":
            params[name] = rng.randint(*spec["range"])
        elif spec["type"] == "float":
            params[name] = round(rng.uniform(*spec["range"]), 4)
        elif spec["type"] == "choice":
            params[name] = rng.choice(spec["options"])
    # Inject GT reproducibility seed (use rng instead of hash() which is
    # non-deterministic across Python processes due to PYTHONHASHSEED)
    params["seed"] = rng.randint(0, 2**31 - 1)
    return params
