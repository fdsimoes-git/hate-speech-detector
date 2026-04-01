import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hate_speech_detector.cli import main


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
