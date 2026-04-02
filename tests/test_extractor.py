from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hate_speech_detector.extractor import extract_audio, extract_audio_from_url


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


# --- extract_audio_from_url tests ---


@patch("hate_speech_detector.extractor.shutil.which", return_value=None)
def test_extract_audio_from_url_ytdlp_not_found(mock_which):
    with pytest.raises(RuntimeError, match="yt-dlp not found"):
        extract_audio_from_url("https://www.youtube.com/watch?v=abc")


@patch("hate_speech_detector.extractor.extract_audio")
@patch("hate_speech_detector.extractor.subprocess.run")
@patch("hate_speech_detector.extractor.shutil.which", return_value="/usr/bin/yt-dlp")
def test_extract_audio_from_url_calls_ytdlp(mock_which, mock_run, mock_extract, tmp_path):
    wav_out = tmp_path / "final.wav"
    mock_extract.return_value = wav_out

    def fake_run(cmd, **kwargs):
        # Simulate yt-dlp creating the output file
        for i, arg in enumerate(cmd):
            if arg == "-o":
                out_template = cmd[i + 1]
                out_file = Path(out_template.replace("%(ext)s", "wav"))
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.touch()
                break
        return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = fake_run

    result = extract_audio_from_url("https://www.youtube.com/watch?v=test")

    assert result == wav_out
    call_args = mock_run.call_args[0][0]
    assert "yt-dlp" in call_args[0]
    assert "--no-playlist" in call_args
    assert "-x" in call_args
    mock_extract.assert_called_once()


@patch("hate_speech_detector.extractor.subprocess.run")
@patch("hate_speech_detector.extractor.shutil.which", return_value="/usr/bin/yt-dlp")
def test_extract_audio_from_url_failure(mock_which, mock_run):
    mock_run.return_value = MagicMock(returncode=1, stderr="Download error")

    with pytest.raises(RuntimeError, match="yt-dlp failed"):
        extract_audio_from_url("https://www.youtube.com/watch?v=bad")
