from pathlib import Path
from unittest.mock import MagicMock, patch

from hate_speech_detector.models import TranscriptSegment
from hate_speech_detector.transcriber import transcribe, _split_long_segment


MOCK_WHISPER_RESULT = {
    "language": "en",
    "segments": [
        {"id": 0, "start": 0.0, "end": 5.5, "text": " Hello world."},
        {"id": 1, "start": 5.5, "end": 12.3, "text": " This is a test segment."},
    ],
}


@patch("hate_speech_detector.transcriber._transcribe_with_whisper", return_value=MOCK_WHISPER_RESULT)
def test_transcribe_returns_segments(mock_whisper):
    segments, lang = transcribe(Path("audio.wav"), model_name="small")

    assert len(segments) == 2
    assert isinstance(segments[0], TranscriptSegment)
    assert segments[0].id == 0
    assert segments[0].start == 0.0
    assert segments[0].end == 5.5
    assert segments[0].text == "Hello world."
    assert segments[1].text == "This is a test segment."
    assert lang == "en"


@patch("hate_speech_detector.transcriber._transcribe_with_whisper", return_value={"segments": []})
def test_transcribe_empty(mock_whisper):
    segments, lang = transcribe(Path("audio.wav"))
    assert segments == []
    assert lang == "unknown"


@patch("hate_speech_detector.transcriber._transcribe_with_whisper", return_value=MOCK_WHISPER_RESULT)
def test_transcribe_passes_language(mock_whisper):
    transcribe(Path("audio.wav"), model_name="small", language="pt")
    mock_whisper.assert_called_once_with(Path("audio.wav"), "small", "pt")


@patch("hate_speech_detector.transcriber._transcribe_with_whisper", return_value=MOCK_WHISPER_RESULT)
def test_transcribe_default_language_is_none(mock_whisper):
    transcribe(Path("audio.wav"), model_name="small")
    mock_whisper.assert_called_once_with(Path("audio.wav"), "small", None)


def test_split_long_segment_short_text():
    seg = TranscriptSegment(id=0, start=0.0, end=5.0, text="Short text.")
    result = _split_long_segment(seg)
    assert len(result) == 1
    assert result[0] is seg


def test_split_long_segment_long_text():
    long_text = ". ".join(f"Sentence number {i}" for i in range(200))
    seg = TranscriptSegment(id=0, start=0.0, end=60.0, text=long_text)
    result = _split_long_segment(seg, max_chars=500)

    assert len(result) > 1
    for chunk in result:
        assert len(chunk.text) <= 600  # allow some tolerance for last sentence
        assert chunk.start >= 0.0
        assert chunk.end <= 60.0
