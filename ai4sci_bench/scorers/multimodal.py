"""Multimodal scorer — image/plot comparison via VLM judge or pixel metrics."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import litellm

from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail
from ai4sci_bench.scorers._judge_common import (
    BACKOFF_BASE,
    DEFAULT_MODEL,
    MAX_RETRIES,
    aggregate_judge_scores,
    evaluator_unavailable_result,
    resolve_model,
    resolve_scorer_api_params,
)
from ai4sci_bench.scorers._parse_utils import parse_judge_json

logger = get_logger(__name__)

VLM_SYSTEM_PROMPT = """\
You are an expert scientific visualization evaluator. You will be shown two images: \
the agent's output plot and a reference plot. Compare them according to the rubric. \
Be fair but rigorous. Return ONLY valid JSON."""

DEFAULT_VLM_RUBRIC = """\
Compare these two scientific plots on a scale of 0 to 10.
Check:
  - Axis labels and ranges (20%)
  - Data/curve shape and key features (40%)
  - Color scheme and visual clarity (20%)
  - Overall similarity to reference (20%)

Return ONLY a JSON object: {"score": <0-10>, "reasoning": "<brief explanation>"}
"""


def _load_image_as_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
    }.get(suffix, "image/png")


@register_scorer("multimodal")
class MultimodalScorer(Scorer):
    """Evaluate visual outputs (plots, figures, heatmaps)."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        mode = config.get("mode", "vlm_judge")
        scorer_name = f"multimodal:{mode}"

        try:
            if mode == "vlm_judge":
                return self._vlm_judge(pred_dir, ref_dir, config, scorer_name)
            elif mode == "pixel_sim":
                return self._pixel_sim(pred_dir, ref_dir, config, scorer_name)
            else:
                return ScoreDetail(
                    scorer_name=scorer_name,
                    score=0.0,
                    max_score=config.get("weight", 1.0),
                    passed=False,
                    details={"error": f"Unknown mode: {mode}"},
                    message=f"Unknown multimodal mode: {mode}",
                )
        except Exception as e:
            logger.error("Multimodal scorer error (mode=%s): %s", mode, e)
            if mode == "vlm_judge":
                return evaluator_unavailable_result(
                    scorer_name=scorer_name,
                    weight=config.get("weight", 1.0),
                    error=e,
                    model=resolve_model(config.get("model", DEFAULT_MODEL)),
                )
            return ScoreDetail(
                scorer_name=scorer_name,
                score=0.0,
                max_score=config.get("weight", 1.0),
                passed=False,
                details={"error": str(e)},
                message=f"Multimodal scorer error: {e}",
            )

    def _vlm_judge(
        self, pred_dir: Path, ref_dir: Path, config: dict, scorer_name: str
    ) -> ScoreDetail:
        weight = config.get("weight", 1.0)
        model = resolve_model(config.get("model", DEFAULT_MODEL))
        rubric = config.get("rubric", DEFAULT_VLM_RUBRIC)
        num_judges = config.get("num_judges", 1)
        temperature = config.get("temperature", 0.0)
        max_tokens = config.get("max_tokens", 1024)
        max_score_value = config.get("max_score_value", 10)
        api_base, api_key = resolve_scorer_api_params(config)
        threshold = config.get("threshold", 0.5)

        if num_judges < 3:
            logger.warning(
                "num_judges=%d provides no parse-failure tolerance; consider >= 3",
                num_judges,
            )

        pred_image_name = config.get("pred_image")
        ref_image_name = config.get("ref_image")

        if not pred_image_name:
            return ScoreDetail(
                scorer_name=scorer_name, score=0.0, max_score=weight, passed=False,
                details={"error": "pred_image not specified in config"},
                message="pred_image not specified",
            )

        pred_path = pred_dir / pred_image_name
        pred_b64 = _load_image_as_base64(pred_path)
        if pred_b64 is None:
            return ScoreDetail(
                scorer_name=scorer_name, score=0.0, max_score=weight, passed=False,
                details={"error": f"Prediction image not found: {pred_path}"},
                message="Prediction image not found",
            )

        ref_b64 = None
        ref_path = None
        if ref_image_name:
            ref_path = ref_dir / ref_image_name
            ref_b64 = _load_image_as_base64(ref_path)

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": f"## Rubric\n{rubric}\n\n## Agent Output Image:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{_guess_mime_type(pred_path)};base64,{pred_b64}"
                },
            },
        ]

        if ref_b64 and ref_path:
            user_content.extend([
                {"type": "text", "text": "## Reference Image:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{_guess_mime_type(ref_path)};base64,{ref_b64}"
                    },
                },
            ])

        user_content.append({
            "type": "text",
            "text": 'Score the agent output. Return ONLY a JSON object: {"score": <number>, "reasoning": "<brief>"}',
        })

        scores: list[float | None] = []
        raw_responses: list[str] = []

        for i in range(num_judges):
            logger.debug("VLM judge %d/%d with model %s", i + 1, num_judges, model)
            raw = self._call_vlm_with_retry(
                model, user_content, temperature, max_tokens,
                api_key=api_key, api_base=api_base,
            )
            raw_responses.append(raw)
            score_val = parse_judge_json(raw, max_score=max_score_value)
            scores.append(score_val)

        result = aggregate_judge_scores(
            scores=scores,
            raw_responses=raw_responses,
            scorer_name=scorer_name,
            model=model,
            num_judges=num_judges,
            max_score_value=max_score_value,
            weight=weight,
            threshold=threshold,
        )

        logger.info(
            "VLM judge result: model=%s, judges=%d, scores=%s, median=%.2f, passed=%s",
            model, num_judges, scores, result.details["median_score"], result.passed,
        )

        return result

    def _call_vlm_with_retry(
        self,
        model: str,
        user_content: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> str:
        last_error: Exception | None = None
        extra_kwargs: dict[str, Any] = {}
        if api_key:
            extra_kwargs["api_key"] = api_key
        elif model.startswith("openrouter/"):
            env_key = os.environ.get("OPENROUTER_API_KEY")
            if env_key:
                extra_kwargs["api_key"] = env_key
        if api_base:
            extra_kwargs["api_base"] = api_base
        for attempt in range(MAX_RETRIES):
            try:
                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": VLM_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra_kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_BASE ** attempt
                    logger.warning(
                        "VLM API call attempt %d failed: %s, retrying in %.1fs",
                        attempt + 1, e, wait,
                    )
                    time.sleep(wait)
        raise last_error  # type: ignore[misc]

    def _pixel_sim(
        self, pred_dir: Path, ref_dir: Path, config: dict, scorer_name: str
    ) -> ScoreDetail:
        import numpy as np

        weight = config.get("weight", 1.0)
        metric = config.get("pixel_metric", "ssim")
        pred_image_name = config.get("pred_image")
        ref_image_name = config.get("ref_image")

        if not pred_image_name or not ref_image_name:
            return ScoreDetail(
                scorer_name=scorer_name, score=0.0, max_score=weight, passed=False,
                details={"error": "pred_image and ref_image must be specified"},
                message="Image files not specified",
            )

        pred_path = pred_dir / pred_image_name
        ref_path = ref_dir / ref_image_name

        if not pred_path.exists():
            return ScoreDetail(
                scorer_name=scorer_name, score=0.0, max_score=weight, passed=False,
                details={"error": f"Prediction image not found: {pred_path}"},
                message="Prediction image not found",
            )
        if not ref_path.exists():
            return ScoreDetail(
                scorer_name=scorer_name, score=0.0, max_score=weight, passed=False,
                details={"error": f"Reference image not found: {ref_path}"},
                message="Reference image not found",
            )

        pred_arr = self._load_image_as_array(pred_path)
        ref_arr = self._load_image_as_array(ref_path)

        if pred_arr is None or ref_arr is None:
            return ScoreDetail(
                scorer_name=scorer_name, score=0.0, max_score=weight, passed=False,
                details={"error": "Failed to load images as arrays"},
                message="Image loading failed",
            )

        if pred_arr.shape != ref_arr.shape:
            return ScoreDetail(
                scorer_name=scorer_name, score=0.0, max_score=weight, passed=False,
                details={
                    "error": "Image shape mismatch",
                    "pred_shape": list(pred_arr.shape),
                    "ref_shape": list(ref_arr.shape),
                },
                message=f"Shape mismatch: {pred_arr.shape} vs {ref_arr.shape}",
            )

        if metric == "ssim":
            similarity = self._compute_ssim(pred_arr, ref_arr)
        elif metric == "psnr":
            similarity = self._compute_psnr(pred_arr, ref_arr)
            similarity = min(similarity / 50.0, 1.0) if similarity > 0 else 0.0
        else:
            return ScoreDetail(
                scorer_name=scorer_name, score=0.0, max_score=weight, passed=False,
                details={"error": f"Unknown pixel_metric: {metric}"},
                message=f"Unknown pixel metric: {metric}",
            )

        final_score = similarity * weight
        threshold = config.get("threshold", 0.8)
        passed = similarity >= threshold

        logger.info(
            "Pixel similarity: metric=%s, value=%.4f, threshold=%.2f, passed=%s",
            metric, similarity, threshold, passed,
        )

        return ScoreDetail(
            scorer_name=scorer_name,
            score=round(final_score, 4),
            max_score=weight,
            passed=passed,
            details={
                "metric": metric,
                "similarity": round(similarity, 4),
                "threshold": threshold,
            },
            message=f"{metric.upper()}: {similarity:.4f}",
        )

    def _load_image_as_array(self, path: Path):
        import numpy as np

        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            return np.array(img, dtype=np.float64) / 255.0
        except ImportError:
            logger.warning("Pillow not installed; pixel_sim mode requires Pillow")
            return None

    def _compute_ssim(self, pred: Any, ref: Any) -> float:
        import numpy as np

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        if pred.ndim == 3:
            ssim_vals = []
            for c in range(pred.shape[2]):
                ssim_vals.append(self._ssim_single_channel(pred[:, :, c], ref[:, :, c], C1, C2))
            return float(np.mean(ssim_vals))
        return self._ssim_single_channel(pred, ref, C1, C2)

    def _ssim_single_channel(self, pred, ref, C1: float, C2: float) -> float:
        import numpy as np

        mu_pred = np.mean(pred)
        mu_ref = np.mean(ref)
        sigma_pred_sq = np.var(pred)
        sigma_ref_sq = np.var(ref)
        sigma_pred_ref = np.mean((pred - mu_pred) * (ref - mu_ref))

        numerator = (2 * mu_pred * mu_ref + C1) * (2 * sigma_pred_ref + C2)
        denominator = (mu_pred**2 + mu_ref**2 + C1) * (sigma_pred_sq + sigma_ref_sq + C2)

        return float(numerator / denominator)

    def _compute_psnr(self, pred: Any, ref: Any) -> float:
        import numpy as np

        mse = np.mean((pred - ref) ** 2)
        if mse == 0:
            return 100.0
        return float(10 * np.log10(1.0 / mse))
