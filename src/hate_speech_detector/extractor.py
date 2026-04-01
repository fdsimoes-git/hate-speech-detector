from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def extract_audio(video_path: Path, output_path: Path | None = None) -> Path:
    """Extract audio from a video file as 16kHz mono WAV (Whisper's expected format)."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with: brew install ffmpeg"
        )

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(tmp.name)
        tmp.close()

    result = subprocess.run(
        [
            "ffmpeg",
            "-i", str(video_path),
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "-y",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    return output_path
