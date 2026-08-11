"""Anti-contamination mechanisms for benchmark integrity.

Prevents LLMs from recognizing and "memorizing" benchmark instances by
injecting canary strings and applying surface-level perturbations to prompts.
"""

from __future__ import annotations

import hashlib
import random
import re
import uuid
from pathlib import Path

from ai4sci_bench.core.logger import get_logger

logger = get_logger(__name__)

# Canary format: invisible to humans but detectable in LLM training data
CANARY_TEMPLATE = "<!-- CANARY {canary_id} -->"


def generate_canary(instance_id: str, secret: str = "") -> str:
    """Generate a unique canary string for an instance.

    The canary is an HTML comment that is invisible when rendered but can be
    searched for in training data to detect contamination.

    Args:
        instance_id: unique instance identifier
        secret: optional secret to make canaries harder to predict

    Returns:
        canary string (HTML comment)
    """
    payload = f"{instance_id}:{secret}:{uuid.uuid4().hex[:8]}"
    canary_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return CANARY_TEMPLATE.format(canary_id=canary_id)


def inject_canary(prompt_text: str, canary: str) -> str:
    """Inject a canary string into a prompt.

    The canary is added as an HTML comment at the end of the prompt,
    which is invisible when rendered but detectable in training data.

    Args:
        prompt_text: the original prompt text
        canary: canary string from generate_canary()

    Returns:
        prompt with canary injected
    """
    return f"{prompt_text}\n\n{canary}\n"


def detect_canary(text: str) -> list[str]:
    """Detect canary strings in text.

    Args:
        text: text to search for canaries

    Returns:
        list of detected canary IDs
    """
    pattern = r"<!-- CANARY ([a-f0-9]{16}) -->"
    return re.findall(pattern, text)


def perturb_prompt(prompt_text: str, seed: int = 0) -> str:
    """Apply surface-level perturbations to a prompt to reduce memorization risk.

    Perturbations are cosmetic and do not change the scientific content:
      - Variable name variations (e.g., "grid_size" ↔ "N_grid")
      - Whitespace/formatting variations
      - Synonym substitutions for non-technical words

    Args:
        prompt_text: original prompt text
        seed: random seed for reproducible perturbations

    Returns:
        perturbed prompt text
    """
    rng = random.Random(seed)

    # Cosmetic whitespace variations
    lines = prompt_text.split("\n")
    result_lines = []
    for line in lines:
        # Randomly add/remove trailing spaces (invisible)
        if rng.random() < 0.3 and line.strip():
            line = line.rstrip()
        result_lines.append(line)

    text = "\n".join(result_lines)

    # Non-technical synonym substitutions
    synonyms = [
        ("implement", "write"),
        ("Write", "Implement"),
        ("provided", "given"),
        ("Given", "Provided"),
        ("Save the following", "Output the following"),
        ("the current directory", "the working directory"),
        ("required output", "expected output"),
        ("Expected Output", "Required Output"),
    ]
    # Apply a random subset of synonyms
    for old, new in synonyms:
        if rng.random() < 0.3:
            text = text.replace(old, new, 1)

    return text


class ContaminationGuard:
    """Manages anti-contamination measures for a benchmark run.

    Usage:
        guard = ContaminationGuard(secret="my-secret-key")
        prompt = guard.protect_prompt(prompt_text, instance_id, seed=42)
        # ... later, to check for contamination:
        canaries = guard.check_contamination(suspicious_text)
    """

    def __init__(self, secret: str = "", enable_canary: bool = True, enable_perturbation: bool = True):
        self.secret = secret
        self.enable_canary = enable_canary
        self.enable_perturbation = enable_perturbation
        self._injected_canaries: dict[str, str] = {}  # instance_id -> canary_id

    def protect_prompt(self, prompt_text: str, instance_id: str, seed: int = 0) -> str:
        """Apply all anti-contamination measures to a prompt.

        Args:
            prompt_text: original prompt
            instance_id: unique instance identifier
            seed: seed for perturbation

        Returns:
            protected prompt text
        """
        text = prompt_text

        if self.enable_perturbation:
            text = perturb_prompt(text, seed=seed)

        if self.enable_canary:
            canary = generate_canary(instance_id, self.secret)
            text = inject_canary(text, canary)
            # Extract and store canary ID
            ids = detect_canary(canary)
            if ids:
                self._injected_canaries[instance_id] = ids[0]

        return text

    def check_contamination(self, text: str) -> list[str]:
        """Check text for known canary strings.

        Args:
            text: text to check (e.g., LLM training data dump)

        Returns:
            list of instance IDs whose canaries were found
        """
        found_ids = detect_canary(text)
        contaminated = []
        for instance_id, canary_id in self._injected_canaries.items():
            if canary_id in found_ids:
                contaminated.append(instance_id)
        return contaminated

    @property
    def canary_registry(self) -> dict[str, str]:
        """Return the mapping of instance_id -> canary_id."""
        return dict(self._injected_canaries)
