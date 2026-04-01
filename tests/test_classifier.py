import sys
from unittest.mock import MagicMock, patch

import pytest

from hate_speech_detector.models import TranscriptSegment
from hate_speech_detector.classifier import CATEGORY_LABELS


NUM_CATEGORIES = len(CATEGORY_LABELS)
LABEL_LIST = list(CATEGORY_LABELS.values())


def _make_nli_result(text, scores=None):
    """Create a single mock NLI pipeline result."""
    if scores is None:
        scores = [0.1] * NUM_CATEGORIES
    return {
        "sequence": text,
        "labels": LABEL_LIST,
        "scores": scores,
    }


@pytest.fixture
def mock_transformers():
    """Mock transformers module so no real model is loaded."""
    mock_tf = MagicMock()
    mock_clf = MagicMock()

    def default_classify(texts, **kwargs):
        if isinstance(texts, str):
            return _make_nli_result(texts)
        return [_make_nli_result(t) for t in texts]

    mock_clf.side_effect = default_classify
    mock_tf.pipeline.return_value = mock_clf

    with patch.dict(sys.modules, {"transformers": mock_tf}):
        yield mock_tf, mock_clf


def _create_classifier(mock_tf, threshold=0.5):
    with patch("hate_speech_detector.classifier._select_device", return_value="cpu"):
        with patch("hate_speech_detector.classifier._clean_apple_double_files"):
            from hate_speech_detector.classifier import HateSpeechClassifier

            return HateSpeechClassifier(threshold=threshold, device="cpu")


def test_classify_returns_correct_count(mock_transformers):
    mock_tf, mock_clf = mock_transformers
    clf = _create_classifier(mock_tf)

    segments = [
        TranscriptSegment(id=0, start=0.0, end=5.0, text="Hello world."),
        TranscriptSegment(id=1, start=5.0, end=10.0, text="Test segment."),
        TranscriptSegment(id=2, start=10.0, end=15.0, text="Another one."),
    ]
    results = clf.classify(segments)
    assert len(results) == 3


def test_classify_categories_sorted_descending(mock_transformers):
    mock_tf, mock_clf = mock_transformers

    # Return varying scores so sorting can be verified
    def varying_classify(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for t in texts:
            scores = [0.1 * (i + 1) for i in range(NUM_CATEGORIES)]
            results.append({"sequence": t, "labels": LABEL_LIST, "scores": scores})
        return results if len(results) > 1 else results[0]

    mock_clf.side_effect = varying_classify
    clf = _create_classifier(mock_tf, threshold=0.0)

    segments = [TranscriptSegment(id=0, start=0.0, end=5.0, text="Some text.")]
    results = clf.classify(segments)

    categories = results[0].categories
    scores = [c.score for c in categories]
    assert scores == sorted(scores, reverse=True)


def test_classify_hate_score_is_max_category(mock_transformers):
    mock_tf, mock_clf = mock_transformers

    def varied_classify(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for t in texts:
            scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55][:NUM_CATEGORIES]
            results.append({"sequence": t, "labels": LABEL_LIST, "scores": scores})
        return results if len(results) > 1 else results[0]

    mock_clf.side_effect = varied_classify
    clf = _create_classifier(mock_tf, threshold=0.0)

    segments = [TranscriptSegment(id=0, start=0.0, end=5.0, text="Test.")]
    results = clf.classify(segments)

    max_cat_score = max(c.score for c in results[0].categories)
    assert abs(results[0].hate_score - max_cat_score) < 1e-6


def test_classify_not_flagged_has_empty_categories(mock_transformers):
    mock_tf, mock_clf = mock_transformers
    # Default mock returns 0.1 for all categories — below threshold 0.5
    clf = _create_classifier(mock_tf, threshold=0.5)

    segments = [TranscriptSegment(id=0, start=0.0, end=5.0, text="Clean text.")]
    results = clf.classify(segments)

    assert not results[0].flagged
    assert results[0].categories == []


def test_classify_all_categories_present(mock_transformers):
    mock_tf, mock_clf = mock_transformers
    clf = _create_classifier(mock_tf, threshold=0.0)

    segments = [TranscriptSegment(id=0, start=0.0, end=5.0, text="Some text.")]
    results = clf.classify(segments)

    category_names = {c.category for c in results[0].categories}
    expected = set(CATEGORY_LABELS.keys())
    assert category_names == expected


def test_classify_flagged_segment(mock_transformers):
    mock_tf, mock_clf = mock_transformers

    def flagged_classify(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for t in texts:
            # First category (racism) scores high
            scores = [0.92] + [0.05] * (NUM_CATEGORIES - 1)
            results.append({"sequence": t, "labels": LABEL_LIST, "scores": scores})
        return results if len(results) > 1 else results[0]

    mock_clf.side_effect = flagged_classify
    clf = _create_classifier(mock_tf, threshold=0.5)

    segments = [TranscriptSegment(id=0, start=0.0, end=5.0, text="Hateful text.")]
    results = clf.classify(segments)

    assert results[0].flagged
    assert results[0].hate_score == pytest.approx(0.92)
    assert len(results[0].categories) == NUM_CATEGORIES
    assert results[0].categories[0].category == "racism"
