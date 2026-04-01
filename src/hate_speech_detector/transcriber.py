from __future__ import annotations

import re
from pathlib import Path

from hate_speech_detector.models import TranscriptSegment


def _split_long_segment(segment: TranscriptSegment, max_chars: int = 1500) -> list[TranscriptSegment]:
    """Split a segment that's too long for the classifier at sentence boundaries."""
    text = segment.text
    if len(text) <= max_chars:
        return [segment]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[TranscriptSegment] = []
    current_text = ""
    chunk_id = 0
    duration = segment.end - segment.start
    chars_total = len(text)

    for sentence in sentences:
        if current_text and len(current_text) + len(sentence) > max_chars:
            # Estimate time boundaries proportionally
            start_ratio = (len(text) - len(text.lstrip())) / max(chars_total, 1)
            progress = len("".join(s.text for s in chunks) + current_text) / max(chars_total, 1)
            chunk_start = segment.start + duration * (progress - len(current_text) / max(chars_total, 1))
            chunk_end = segment.start + duration * progress

            chunks.append(TranscriptSegment(
                id=segment.id * 1000 + chunk_id,
                start=chunk_start,
                end=chunk_end,
                text=current_text.strip(),
            ))
            current_text = sentence
            chunk_id += 1
        else:
            current_text = f"{current_text} {sentence}".strip() if current_text else sentence

    if current_text.strip():
        progress = 1.0
        prev_progress = len("".join(s.text for s in chunks)) / max(chars_total, 1)
        chunks.append(TranscriptSegment(
            id=segment.id * 1000 + chunk_id,
            start=segment.start + duration * prev_progress,
            end=segment.end,
            text=current_text.strip(),
        ))

    return chunks if chunks else [segment]


def transcribe(audio_path: Path, model_name: str = "small", language: str | None = None) -> list[TranscriptSegment]:
    """Transcribe audio to timestamped segments using Whisper.

    Tries mlx-whisper (Apple Silicon optimized) first, falls back to openai-whisper.
    """
    result = _transcribe_with_whisper(audio_path, model_name, language)

    segments: list[TranscriptSegment] = []
    for raw in result.get("segments", []):
        seg = TranscriptSegment(
            id=raw["id"],
            start=raw["start"],
            end=raw["end"],
            text=raw["text"].strip(),
        )
        segments.extend(_split_long_segment(seg))

    return segments


def _transcribe_with_whisper(audio_path: Path, model_name: str, language: str | None = None) -> dict:
    """Run Whisper transcription, preferring mlx-whisper on Apple Silicon."""
    try:
        import mlx_whisper

        return mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=f"mlx-community/whisper-{model_name}-mlx",
            language=language,
        )
    except ImportError:
        pass

    import whisper

    model = whisper.load_model(model_name)
    return model.transcribe(str(audio_path), language=language)
