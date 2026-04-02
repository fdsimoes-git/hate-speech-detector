import json

from rich.text import Text

from hate_speech_detector.reporter import _build_timeline, _format_time, _score_style, print_report, write_json


def test_format_time_zero():
    assert _format_time(0.0) == "00:00:00"


def test_format_time_seconds():
    assert _format_time(65.5) == "00:01:05"


def test_format_time_hours():
    assert _format_time(3661.0) == "01:01:01"


def test_score_style_high():
    assert "red" in _score_style(0.8)


def test_score_style_medium():
    assert _score_style(0.5) == "yellow"


def test_score_style_low():
    assert _score_style(0.2) == "green"


def test_print_report_no_flags(sample_report, capsys):
    for c in sample_report.classifications:
        c.flagged = False
        c.categories = []
    sample_report.segments_flagged = 0

    print_report(sample_report)
    output = capsys.readouterr().out

    assert "No hate speech detected" in output


def test_print_report_with_flags(sample_report, capsys):
    print_report(sample_report)
    output = capsys.readouterr().out

    assert "Flagged Segments (1)" in output
    assert "00:00:05" in output
    assert "00:00:12" in output
    assert "racism" in output
    assert "0.73" in output


def test_print_report_verbose(sample_report, capsys):
    print_report(sample_report, verbose=True)
    output = capsys.readouterr().out

    assert "Flagged Segments" in output
    assert "Clean Segments" in output
    assert "Hello and welcome" in output


def test_print_report_summary(sample_report, capsys):
    print_report(sample_report)
    output = capsys.readouterr().out

    assert "1 of 3 segments flagged" in output


def test_build_timeline_returns_text(sample_report):
    result = _build_timeline(sample_report, width=60)
    assert isinstance(result, Text)
    plain = result.plain
    assert "00:00" in plain
    assert "00:00:18" in plain


def test_print_report_includes_timeline(sample_report, capsys):
    print_report(sample_report)
    output = capsys.readouterr().out
    assert "Timeline" in output


def test_write_json(sample_report, tmp_path):
    output_path = tmp_path / "report.json"
    write_json(sample_report, output_path)

    with open(output_path) as f:
        data = json.load(f)

    assert data["source_file"] == "test_video.mp4"
    assert data["segments_total"] == 3
    assert data["segments_flagged"] == 1
    assert len(data["classifications"]) == 3
    assert data["classifications"][1]["flagged"] is True
    assert data["classifications"][1]["categories"][0]["category"] == "racism"
