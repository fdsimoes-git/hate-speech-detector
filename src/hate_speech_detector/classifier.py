from __future__ import annotations

import platform
import shutil
import subprocess
import sys

from hate_speech_detector.models import (
    CategoryScore,
    SegmentClassification,
    TranscriptSegment,
)

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

CATEGORY_LABELS: dict[str, str] = {
    "racism": "racist speech, racial discrimination, or dehumanization based on race or ethnicity",
    "sexism": "sexist speech, misogyny, or gender-based discrimination",
    "homophobia": "homophobic speech or discrimination against LGBTQ people",
    "religious_intolerance": "religious intolerance or hatred toward religious groups",
    "ableism": "ableist speech or discrimination against people with disabilities",
    "xenophobia": "xenophobic speech or hatred toward immigrants and foreigners",
}

HYPOTHESIS_TEMPLATE = "This text contains {}."


def _clean_apple_double_files() -> None:
    """Remove macOS Apple Double (._*) files from transformers package.

    On external/encrypted APFS volumes, macOS creates ._* resource fork files
    that can cause UnicodeDecodeError when packages scan their own directories.
    """
    if platform.system() != "Darwin" or not shutil.which("dot_clean"):
        return
    try:
        import transformers

        pkg_dir = str(transformers.__path__[0])
        subprocess.run(["dot_clean", pkg_dir], capture_output=True, timeout=30)
    except Exception:
        pass


def _select_device(preferred: str) -> str:
    if preferred == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        print("MPS not available, falling back to CPU", file=sys.stderr)
        return "cpu"
    return preferred


class HateSpeechClassifier:
    def __init__(self, threshold: float = 0.5, device: str = "mps", batch_size: int = 8):
        self.threshold = threshold
        self.batch_size = batch_size

        _clean_apple_double_files()

        from transformers import pipeline as transformers_pipeline

        resolved_device = _select_device(device)

        self._classifier = transformers_pipeline(
            "zero-shot-classification",
            model=NLI_MODEL,
            device=resolved_device,
        )

        self._categories = list(CATEGORY_LABELS.keys())
        self._labels = list(CATEGORY_LABELS.values())
        self._label_to_category = {label: cat for cat, label in CATEGORY_LABELS.items()}

    def classify(self, segments: list[TranscriptSegment]) -> list[SegmentClassification]:
        texts = [seg.text for seg in segments]
        total = len(segments)

        # Build context window: prev + current + next segment
        contextualized = []
        for i, text in enumerate(texts):
            parts = []
            if i > 0:
                parts.append(texts[i - 1])
            parts.append(text)
            if i < total - 1:
                parts.append(texts[i + 1])
            contextualized.append(" ".join(parts))

        nli_results = self._classifier(
            contextualized,
            candidate_labels=self._labels,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True,
            batch_size=self.batch_size,
        )

        # Single result comes as a dict, not a list
        if isinstance(nli_results, dict):
            nli_results = [nli_results]

        results: list[SegmentClassification] = []
        for segment, nli_result, ctx_text in zip(segments, nli_results, contextualized):
            cat_scores: dict[str, float] = {}
            for label, score in zip(nli_result["labels"], nli_result["scores"]):
                cat_name = self._label_to_category[label]
                cat_scores[cat_name] = score

            hate_score = max(cat_scores.values())
            flagged = hate_score >= self.threshold

            categories = sorted(
                [CategoryScore(category=cat, score=score) for cat, score in cat_scores.items()],
                key=lambda c: c.score,
                reverse=True,
            )

            # Store context only when it differs from the segment text
            context = ctx_text if ctx_text != segment.text else ""

            results.append(
                SegmentClassification(
                    segment=segment,
                    hate_score=hate_score,
                    flagged=flagged,
                    categories=categories if flagged else [],
                    context=context,
                )
            )

        return results
