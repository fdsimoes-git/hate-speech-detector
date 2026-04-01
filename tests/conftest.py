from hate_speech_detector.models import (
    AnalysisReport,
    CategoryScore,
    SegmentClassification,
    TranscriptSegment,
)

import pytest


@pytest.fixture
def sample_segments():
    return [
        TranscriptSegment(id=0, start=0.0, end=5.0, text="Hello and welcome to the show."),
        TranscriptSegment(id=1, start=5.0, end=12.0, text="This is a hateful statement targeting a group."),
        TranscriptSegment(id=2, start=12.0, end=18.0, text="The weather is nice today."),
    ]


@pytest.fixture
def sample_classifications(sample_segments):
    return [
        SegmentClassification(
            segment=sample_segments[0], hate_score=0.05, flagged=False, categories=[]
        ),
        SegmentClassification(
            segment=sample_segments[1],
            hate_score=0.92,
            flagged=True,
            categories=[
                CategoryScore(category="racism", score=0.73),
                CategoryScore(category="sexism", score=0.12),
                CategoryScore(category="religious_intolerance", score=0.08),
            ],
        ),
        SegmentClassification(
            segment=sample_segments[2], hate_score=0.03, flagged=False, categories=[]
        ),
    ]


@pytest.fixture
def sample_report(sample_classifications):
    return AnalysisReport(
        source_file="test_video.mp4",
        duration_seconds=18.0,
        whisper_model="small",
        segments_total=3,
        segments_flagged=1,
        classifications=sample_classifications,
        generated_at="2026-04-01T00:00:00+00:00",
    )
