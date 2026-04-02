import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from hate_speech_detector.models import TranscriptSegment
from hate_speech_detector.classifier import CATEGORY_REFERENCES


NUM_CATEGORIES = len(CATEGORY_REFERENCES)


def _mock_encode(texts, **kwargs):
    """Return deterministic normalized embeddings."""
    n = len(texts) if isinstance(texts, list) else 1
    emb = torch.arange(n * 768, dtype=torch.float32).reshape(n, 768)
    emb = emb / emb.norm(dim=1, keepdim=True)
    if kwargs.get("convert_to_tensor"):
        return emb
    return emb.numpy()


@pytest.fixture
def mock_sentence_transformers():
    """Mock sentence_transformers module."""
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.side_effect = _mock_encode
    mock_st.SentenceTransformer.return_value = mock_model

    # Mock util.cos_sim to use the real implementation
    mock_st.util.cos_sim = lambda a, b: torch.mm(a, b.T)

    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        yield mock_st, mock_model


def _create_classifier(mock_st, threshold=0.20):
    with patch("hate_speech_detector.classifier._select_device", return_value="cpu"):
        with patch("hate_speech_detector.classifier._clean_apple_double_files"):
            from hate_speech_detector.classifier import HateSpeechClassifier

            return HateSpeechClassifier(threshold=threshold, device="cpu")


def test_classify_returns_correct_count(mock_sentence_transformers):
    mock_st, mock_model = mock_sentence_transformers
    clf = _create_classifier(mock_st)

    segments = [
        TranscriptSegment(id=0, start=0.0, end=5.0, text="Hello world."),
        TranscriptSegment(id=1, start=5.0, end=10.0, text="Test segment."),
        TranscriptSegment(id=2, start=10.0, end=15.0, text="Another one."),
    ]
    results = clf.classify(segments)
    assert len(results) == 3


def test_classify_categories_sorted_descending(mock_sentence_transformers):
    mock_st, mock_model = mock_sentence_transformers
    clf = _create_classifier(mock_st, threshold=0.0)  # flag everything

    segments = [TranscriptSegment(id=0, start=0.0, end=5.0, text="Some text.")]
    results = clf.classify(segments)

    categories = results[0].categories
    scores = [c.score for c in categories]
    assert scores == sorted(scores, reverse=True)


def test_classify_hate_score_is_max_similarity(mock_sentence_transformers):
    mock_st, mock_model = mock_sentence_transformers
    clf = _create_classifier(mock_st, threshold=0.0)

    segments = [TranscriptSegment(id=0, start=0.0, end=5.0, text="Test.")]
    results = clf.classify(segments)

    max_cat_score = max(c.score for c in results[0].categories)
    assert abs(results[0].hate_score - max_cat_score) < 1e-6


def test_classify_not_flagged_has_empty_categories(mock_sentence_transformers):
    mock_st, mock_model = mock_sentence_transformers

    # Make segment embedding orthogonal to all category embeddings
    call_count = [0]

    def orthogonal_encode(texts, **kwargs):
        call_count[0] += 1
        n = len(texts) if isinstance(texts, list) else 1
        if call_count[0] == 1:
            emb = torch.eye(768)[:n]
        else:
            emb = torch.zeros(n, 768)
            emb[:, 767] = 1.0
        return emb

    mock_model.encode.side_effect = orthogonal_encode
    clf = _create_classifier(mock_st, threshold=0.20)

    segments = [TranscriptSegment(id=0, start=0.0, end=5.0, text="Clean text.")]
    results = clf.classify(segments)

    assert not results[0].flagged
    assert results[0].categories == []


def test_classify_all_categories_present(mock_sentence_transformers):
    mock_st, mock_model = mock_sentence_transformers
    clf = _create_classifier(mock_st, threshold=0.0)

    segments = [TranscriptSegment(id=0, start=0.0, end=5.0, text="Some text.")]
    results = clf.classify(segments)

    category_names = {c.category for c in results[0].categories}
    expected = set(CATEGORY_REFERENCES.keys())
    assert category_names == expected


def test_classify_context_stored_for_middle_segment(mock_sentence_transformers):
    """Middle segments should have context from neighbors."""
    mock_st, mock_model = mock_sentence_transformers
    clf = _create_classifier(mock_st, threshold=0.0)

    segments = [
        TranscriptSegment(id=0, start=0.0, end=5.0, text="First."),
        TranscriptSegment(id=1, start=5.0, end=10.0, text="Middle."),
        TranscriptSegment(id=2, start=10.0, end=15.0, text="Last."),
    ]
    results = clf.classify(segments)

    # Middle segment should have context with neighbors
    assert results[1].context == "First. Middle. Last."
    # First and last have partial context
    assert "First." in results[0].context
    assert "Last." in results[2].context
