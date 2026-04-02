"""Core analysis pipeline — shared by CLI and server."""
from __future__ import annotations

import gc
from pathlib import Path

from hate_speech_detector.extractor import extract_audio, extract_audio_from_url
from hate_speech_detector.models import AnalysisReport


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def analyze(
    source: str,
    *,
    model: str = "small",
    language: str | None = None,
    threshold: float = 0.20,
    device: str = "mps",
    verify: bool = False,
    api_key: str | None = None,
    custom_references: dict[str, list[str]] | None = None,
) -> AnalysisReport:
    """Run the full analysis pipeline on a video file or URL.

    Returns an AnalysisReport. The caller is responsible for presentation.
    """
    # Step 1: Extract audio
    if _is_url(source):
        audio_path = extract_audio_from_url(source)
    else:
        audio_path = extract_audio(Path(source))

    try:
        # Step 2: Transcribe
        from hate_speech_detector.transcriber import transcribe

        segments = transcribe(audio_path, model_name=model, language=language)

        gc.collect()

        # Step 3: Classify
        from hate_speech_detector.classifier import HateSpeechClassifier

        classifier = HateSpeechClassifier(
            threshold=threshold,
            device=device,
            custom_references=custom_references,
        )
        classifications = classifier.classify(segments)

        # Step 4: LLM verification
        if verify:
            from hate_speech_detector.llm_verifier import verify_segments

            classifications = verify_segments(classifications, api_key=api_key)

        # Step 5: Build report
        duration = max((s.end for s in segments), default=0.0)
        flagged_count = sum(1 for c in classifications if c.flagged)

        return AnalysisReport(
            source_file=source,
            duration_seconds=duration,
            whisper_model=model,
            segments_total=len(segments),
            segments_flagged=flagged_count,
            classifications=classifications,
        )
    finally:
        try:
            audio_path.unlink()
        except OSError:
            pass
