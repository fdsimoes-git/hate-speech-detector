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

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CATEGORY_REFERENCES: dict[str, list[str]] = {
    "racism": [
        "racism, racial discrimination, racial hatred against ethnic groups",
        "comparing people to animals based on race, dehumanizing people, treating minorities as subhuman",
        "stereotyping indigenous people as lazy, primitive, or worthless",
        "complaining about land rights of racial minorities, quilombola communities, or indigenous reserves as a waste",
        "using livestock or animal terminology to describe people of a certain race",
        "measuring, weighing, or quantifying people as if they were animals or property",
        "mocking people's physical features or bodies based on their race or ethnicity",
    ],
    "sexism": [
        "sexism, misogyny, gender discrimination, degrading women",
        "objectifying women, reducing women to their appearance or bodies",
        "claiming women are inferior, less capable, or belong in domestic roles",
    ],
    "homophobia": [
        "homophobia, anti-LGBTQ hatred, discrimination based on sexual orientation",
        "mocking or threatening gay, lesbian, bisexual, or transgender people",
        "claiming homosexuality is a disease, sin, or abnormality",
    ],
    "religious_intolerance": [
        "religious intolerance, hatred towards religious groups",
        "attacking people for their faith, mocking religious practices",
        "blaming social problems on a specific religion or its followers",
    ],
    "ableism": [
        "ableism, discrimination against disabled people, mocking disabilities",
        "calling people retarded, crippled, or using disability as an insult",
    ],
    "xenophobia": [
        "xenophobia, hatred towards immigrants and foreigners",
        "blaming immigrants for crime, unemployment, or cultural decline",
        "calling for deportation or exclusion of foreign-born people",
        "claiming a nation has been corrupted, weakened, or sold out by outsiders",
    ],
}


def _clean_apple_double_files() -> None:
    """Remove macOS Apple Double (._*) files from sentence_transformers package.

    On external/encrypted APFS volumes, macOS creates ._* resource fork files
    that can cause UnicodeDecodeError when packages scan their own directories.
    """
    if platform.system() != "Darwin" or not shutil.which("dot_clean"):
        return
    try:
        import sentence_transformers

        pkg_dir = str(sentence_transformers.__path__[0])
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


def _merge_references(
    base: dict[str, list[str]], custom: dict[str, list[str]] | None
) -> dict[str, list[str]]:
    """Merge custom reference texts into the base references.

    Custom references extend existing categories or add new ones.
    """
    if not custom:
        return base
    merged = {cat: list(refs) for cat, refs in base.items()}
    for cat, refs in custom.items():
        if cat in merged:
            merged[cat].extend(refs)
        else:
            merged[cat] = list(refs)
    return merged


class HateSpeechClassifier:
    def __init__(
        self,
        threshold: float = 0.20,
        device: str = "mps",
        batch_size: int = 16,
        custom_references: dict[str, list[str]] | None = None,
    ):
        self.threshold = threshold
        self.batch_size = batch_size

        _clean_apple_double_files()
        from sentence_transformers import SentenceTransformer

        resolved_device = _select_device(device)

        self._model = SentenceTransformer(MODEL_NAME, device=resolved_device)

        # Merge built-in + custom references, then flatten
        references = _merge_references(CATEGORY_REFERENCES, custom_references)
        self._categories = list(references.keys())
        self._ref_categories: list[str] = []
        all_refs: list[str] = []
        for cat, refs in references.items():
            for ref in refs:
                self._ref_categories.append(cat)
                all_refs.append(ref)

        self._ref_embeddings = self._model.encode(
            all_refs, convert_to_tensor=True, normalize_embeddings=True
        )

    def classify(self, segments: list[TranscriptSegment]) -> list[SegmentClassification]:
        from sentence_transformers import util

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

        # Encode contextualized segment texts
        segment_embeddings = self._model.encode(
            contextualized,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Cosine similarity: (num_segments, num_all_references)
        similarity_matrix = util.cos_sim(segment_embeddings, self._ref_embeddings)

        results: list[SegmentClassification] = []
        for i, segment in enumerate(segments):
            # For each category, take max similarity across its reference embeddings
            cat_scores: dict[str, float] = {}
            for j, cat in enumerate(self._ref_categories):
                score = float(similarity_matrix[i][j])
                if cat not in cat_scores or score > cat_scores[cat]:
                    cat_scores[cat] = score

            hate_score = max(cat_scores.values())
            flagged = hate_score >= self.threshold

            categories = sorted(
                [CategoryScore(category=cat, score=score) for cat, score in cat_scores.items()],
                key=lambda c: c.score,
                reverse=True,
            )

            # Store context only when it differs from the segment text
            ctx_text = contextualized[i]
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
