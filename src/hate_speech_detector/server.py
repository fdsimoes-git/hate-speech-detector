"""FastAPI server — exposes the analysis pipeline as an HTTP API."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from hate_speech_detector.pipeline import analyze

app = FastAPI(
    title="hate-speech-detector",
    description="Analyze video/audio files for hate speech content.",
    version="0.3.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    model: str = Form("small"),
    language: str | None = Form(None),
    threshold: float = Form(0.20),
    verify: bool = Form(False),
    api_key: str | None = Form(None),
):
    """Analyze a video for hate speech.

    Send either a file upload OR a URL (e.g. YouTube), not both.
    """
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide either a file upload or a url.")
    if file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a url, not both.")

    device = os.environ.get("HSD_DEVICE", "mps")
    source: str

    tmp_path: Path | None = None
    try:
        if url:
            source = url
        else:
            # Save upload to temp file
            suffix = Path(file.filename or "video").suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_path = Path(tmp.name)
            tmp.write(await file.read())
            tmp.close()
            source = str(tmp_path)

        report = analyze(
            source=source,
            model=model,
            language=language,
            threshold=threshold,
            device=device,
            verify=verify,
            api_key=api_key,
        )

        return JSONResponse(content=report.to_dict())

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path:
            try:
                tmp_path.unlink()
            except OSError:
                pass
