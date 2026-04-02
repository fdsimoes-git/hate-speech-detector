from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from hate_speech_detector.models import AnalysisReport
from hate_speech_detector.server import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_no_input():
    response = client.post("/analyze")
    assert response.status_code == 400


def test_analyze_both_file_and_url():
    response = client.post(
        "/analyze",
        data={"url": "https://youtube.com/watch?v=test"},
        files={"file": ("video.mp4", b"fake", "video/mp4")},
    )
    assert response.status_code == 400


@patch("hate_speech_detector.server.analyze")
def test_analyze_url(mock_analyze):
    mock_analyze.return_value = AnalysisReport(
        source_file="https://youtube.com/watch?v=test",
        duration_seconds=10.0,
        whisper_model="small",
        segments_total=2,
        segments_flagged=0,
        classifications=[],
    )

    response = client.post("/analyze", data={"url": "https://youtube.com/watch?v=test"})

    assert response.status_code == 200
    data = response.json()
    assert data["segments_total"] == 2
    assert data["segments_flagged"] == 0
    mock_analyze.assert_called_once()


@patch("hate_speech_detector.server.analyze")
def test_analyze_file_upload(mock_analyze):
    mock_analyze.return_value = AnalysisReport(
        source_file="video.mp4",
        duration_seconds=5.0,
        whisper_model="small",
        segments_total=1,
        segments_flagged=0,
        classifications=[],
    )

    response = client.post(
        "/analyze",
        files={"file": ("video.mp4", b"fake-video-data", "video/mp4")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["segments_total"] == 1
    mock_analyze.assert_called_once()
