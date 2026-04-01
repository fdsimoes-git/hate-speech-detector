from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hate_speech_detector.extractor import extract_audio


def test_extract_audio_file_not_found():
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        extract_audio(Path("/nonexistent/video.mp4"))


@patch("hate_speech_detector.extractor.shutil.which", return_value=None)
def test_extract_audio_ffmpeg_not_found(mock_which, tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        extract_audio(video)


@patch("hate_speech_detector.extractor.subprocess.run")
@patch("hate_speech_detector.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
def test_extract_audio_calls_ffmpeg(mock_which, mock_run, tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "output.wav"

    mock_run.return_value = MagicMock(returncode=0, stderr="")

    result = extract_audio(video, output_path=output)

    assert result == output
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "ffmpeg"
    assert "-ar" in call_args
    assert "16000" in call_args
    assert "-ac" in call_args
    assert "1" in call_args


@patch("hate_speech_detector.extractor.subprocess.run")
@patch("hate_speech_detector.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
def test_extract_audio_ffmpeg_failure(mock_which, mock_run, tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()

    mock_run.return_value = MagicMock(returncode=1, stderr="Encoding error")

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        extract_audio(video)


@patch("hate_speech_detector.extractor.subprocess.run")
@patch("hate_speech_detector.extractor.shutil.which", return_value="/usr/bin/ffmpeg")
def test_extract_audio_temp_file(mock_which, mock_run, tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    mock_run.return_value = MagicMock(returncode=0, stderr="")

    result = extract_audio(video)

    assert result.suffix == ".wav"
