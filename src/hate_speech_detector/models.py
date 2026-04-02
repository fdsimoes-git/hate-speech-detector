from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "start": self.start, "end": self.end, "text": self.text}


@dataclass
class CategoryScore:
    category: str
    score: float

    def to_dict(self) -> dict:
        return {"category": self.category, "score": self.score}


@dataclass
class SegmentClassification:
    segment: TranscriptSegment
    hate_score: float
    flagged: bool
    categories: list[CategoryScore] = field(default_factory=list)
    context: str = ""
    reasoning: str = ""
    embedding_score: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "segment": self.segment.to_dict(),
            "hate_score": self.hate_score,
            "flagged": self.flagged,
            "categories": [c.to_dict() for c in self.categories],
        }
        if self.context:
            d["context"] = self.context
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.embedding_score > 0:
            d["embedding_score"] = self.embedding_score
        return d


@dataclass
class AnalysisReport:
    source_file: str
    duration_seconds: float
    whisper_model: str
    segments_total: int
    segments_flagged: int
    classifications: list[SegmentClassification]
    language: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        d = {
            "source_file": self.source_file,
            "duration_seconds": self.duration_seconds,
            "whisper_model": self.whisper_model,
            "segments_total": self.segments_total,
            "segments_flagged": self.segments_flagged,
            "classifications": [c.to_dict() for c in self.classifications],
            "generated_at": self.generated_at,
        }
        if self.language:
            d["language"] = self.language
        return d
