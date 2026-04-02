from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hate_speech_detector.llm_verifier import (
    PREFILTER_THRESHOLD,
    _build_prompt,
    _parse_response,
    verify_segments,
)
from hate_speech_detector.models import (
    CategoryScore,
    SegmentClassification,
    TranscriptSegment,
)


@pytest.fixture
def segments():
    return [
        TranscriptSegment(id=0, start=0.0, end=5.0, text="Hello everyone."),
        TranscriptSegment(id=1, start=5.0, end=10.0, text="A hateful statement."),
        TranscriptSegment(id=2, start=10.0, end=15.0, text="Nice weather today."),
    ]


@pytest.fixture
def classifications(segments):
    """Segment 0 below threshold, segment 1 above, segment 2 below."""
    return [
        SegmentClassification(
            segment=segments[0], hate_score=0.05, flagged=False, categories=[]
        ),
        SegmentClassification(
            segment=segments[1],
            hate_score=0.45,
            flagged=True,
            categories=[CategoryScore(category="racism", score=0.45)],
            context="Hello everyone. A hateful statement. Nice weather today.",
        ),
        SegmentClassification(
            segment=segments[2], hate_score=0.08, flagged=False, categories=[]
        ),
    ]


class TestParseResponse:
    def test_plain_json(self):
        text = '[{"segment_id": 1, "flagged": true, "hate_score": 0.8}]'
        result = _parse_response(text)
        assert len(result) == 1
        assert result[0]["flagged"] is True

    def test_json_in_code_fence(self):
        text = '```json\n[{"segment_id": 1, "flagged": false}]\n```'
        result = _parse_response(text)
        assert result[0]["flagged"] is False

    def test_whitespace_padding(self):
        text = '  \n[{"segment_id": 0}]\n  '
        result = _parse_response(text)
        assert result[0]["segment_id"] == 0


class TestBuildPrompt:
    def test_includes_segment_text(self, classifications):
        candidates = [(1, classifications[1])]
        prompt = _build_prompt(candidates)
        assert "A hateful statement." in prompt
        assert "Segment 1" in prompt

    def test_includes_context(self, classifications):
        candidates = [(1, classifications[1])]
        prompt = _build_prompt(candidates)
        assert "Context:" in prompt


class TestVerifySegments:
    def _make_verdicts_json(self, verdicts: list[dict]) -> str:
        return json.dumps(verdicts)

    @patch("hate_speech_detector.llm_verifier._call_cli")
    def test_below_threshold_untouched(self, mock_call_cli, segments):
        """All segments below threshold → no CLI call, returned as-is."""
        clfs = [
            SegmentClassification(
                segment=s, hate_score=0.01, flagged=False, categories=[]
            )
            for s in segments
        ]
        result = verify_segments(clfs)
        assert len(result) == 3
        mock_call_cli.assert_not_called()

    @patch("hate_speech_detector.llm_verifier._call_cli")
    def test_cli_verdict_applied(self, mock_call_cli, classifications):
        """Segment above threshold gets CLI verdict applied."""
        verdicts = [
            {
                "segment_id": 1,
                "flagged": True,
                "hate_score": 0.85,
                "categories": [{"category": "racism", "score": 0.85}],
                "reasoning": "Dehumanizing language targeting a group.",
            }
        ]
        mock_call_cli.return_value = self._make_verdicts_json(verdicts)

        result = verify_segments(classifications)

        assert len(result) == 3
        # Segment 1 should have LLM verdict
        assert result[1].hate_score == 0.85
        assert result[1].flagged is True
        assert result[1].reasoning == "Dehumanizing language targeting a group."
        assert result[1].embedding_score == 0.45  # original embedding score
        assert len(result[1].categories) == 1
        assert result[1].categories[0].category == "racism"

        # Segment 0 and 2 should be unchanged (below threshold)
        assert result[0].hate_score == 0.05
        assert result[0].flagged is False
        assert result[2].hate_score == 0.08
        assert result[2].flagged is False

    @patch("hate_speech_detector.llm_verifier._call_cli")
    def test_cli_clears_false_positive(self, mock_call_cli, classifications):
        """LLM can mark a pre-filtered candidate as not flagged."""
        verdicts = [
            {
                "segment_id": 1,
                "flagged": False,
                "hate_score": 0.1,
                "categories": [],
                "reasoning": "Political commentary, not hate speech.",
            }
        ]
        mock_call_cli.return_value = self._make_verdicts_json(verdicts)

        result = verify_segments(classifications)

        assert result[1].flagged is False
        assert result[1].hate_score == 0.1
        assert result[1].categories == []
        assert result[1].reasoning == "Political commentary, not hate speech."

    @patch("hate_speech_detector.llm_verifier._call_api")
    def test_api_key_uses_direct_api(self, mock_call_api, classifications):
        """When api_key is provided, uses the direct API instead of CLI."""
        verdicts = [
            {
                "segment_id": 1,
                "flagged": True,
                "hate_score": 0.80,
                "categories": [{"category": "racism", "score": 0.80}],
                "reasoning": "Hateful.",
            }
        ]
        mock_call_api.return_value = self._make_verdicts_json(verdicts)

        result = verify_segments(classifications, api_key="sk-ant-test")

        mock_call_api.assert_called_once()
        assert result[1].flagged is True
        assert result[1].hate_score == 0.80

    @patch("hate_speech_detector.llm_verifier._call_api")
    def test_api_key_from_env(self, mock_call_api, classifications):
        """ANTHROPIC_API_KEY env var triggers direct API mode."""
        verdicts = [
            {
                "segment_id": 1,
                "flagged": True,
                "hate_score": 0.70,
                "categories": [],
                "reasoning": "Test.",
            }
        ]
        mock_call_api.return_value = self._make_verdicts_json(verdicts)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env"}):
            verify_segments(classifications)

        mock_call_api.assert_called_once()
