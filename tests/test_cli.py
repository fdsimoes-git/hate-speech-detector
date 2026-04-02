import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hate_speech_detector.cli import _is_url, main


def test_is_url():
    assert _is_url("https://www.youtube.com/watch?v=abc") is True
    assert _is_url("http://example.com/video") is True
    assert _is_url("/path/to/video.mp4") is False
    assert _is_url("video.mp4") is False


def test_cli_url_missing_ytdlp(capsys):
    with patch("hate_speech_detector.cli.shutil.which", side_effect=lambda x: None if x == "yt-dlp" else "/usr/bin/" + x):
        with patch("sys.argv", ["hate-speech-detector", "https://youtube.com/watch?v=test"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


def test_cli_missing_file(capsys):
    with patch("sys.argv", ["hate-speech-detector", "/nonexistent/video.mp4"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_cli_invalid_threshold(capsys, tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    with patch("sys.argv", ["hate-speech-detector", str(video), "--threshold", "1.5"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
