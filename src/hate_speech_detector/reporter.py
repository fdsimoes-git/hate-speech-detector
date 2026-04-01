from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import box

from hate_speech_detector.models import AnalysisReport, SegmentClassification


def _format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _score_style(score: float) -> str:
    """Return a Rich style string based on score severity."""
    if score >= 0.7:
        return "bold red"
    if score >= 0.4:
        return "yellow"
    return "green"


def _build_segment_text(classification: SegmentClassification) -> Text:
    """Build a Rich Text block for a single segment's details."""
    seg = classification.segment
    text = Text()

    # Quoted segment text
    text.append(f'"{seg.text}"', style="bold")

    # Context if flagged
    if classification.context and classification.flagged:
        text.append("\n\n")
        text.append("Scored with context:\n", style="dim")
        text.append(f'"{classification.context}"', style="dim italic")

    # Category scores with colored bars
    if classification.categories:
        text.append("\n\n")
        max_name_len = max(len(c.category) for c in classification.categories)
        for i, cat in enumerate(classification.categories):
            name = cat.category.ljust(max_name_len)
            style = _score_style(cat.score)
            text.append(f"  {name}  ", style="dim")
            text.append(f"{cat.score:.2f} ", style=style)
            filled = int(cat.score * 20)
            bar_color = "red" if cat.score >= 0.7 else "yellow" if cat.score >= 0.4 else "green"
            text.append("\u2588" * filled, style=bar_color)
            text.append("\u2591" * (20 - filled), style="dim")
            if i < len(classification.categories) - 1:
                text.append("\n")

    return text


def _print_segment_panel(
    console: Console, classification: SegmentClassification, *, dim: bool = False
) -> None:
    """Print a single segment as a Rich Panel."""
    seg = classification.segment
    score = classification.hate_score
    time_range = f"{_format_time(seg.start)} \u2192 {_format_time(seg.end)}"

    if dim:
        border = "dim"
        title_style = "dim"
        subtitle_style = "green"
    else:
        border = _score_style(score)
        title_style = "bold"
        subtitle_style = _score_style(score)

    content = _build_segment_text(classification)

    panel = Panel(
        content,
        title=f"[{title_style}]{time_range}[/{title_style}]",
        subtitle=f"[{subtitle_style}]score: {score:.2f}[/{subtitle_style}]",
        border_style=border,
        padding=(1, 2),
    )
    console.print(panel)


def print_report(report: AnalysisReport, verbose: bool = False) -> None:
    """Print a formatted report to the terminal using Rich."""
    console = Console(highlight=False)

    # Title
    console.print()
    console.print(
        Panel(
            "[bold]Hate Speech Analysis Report[/bold]",
            box=box.DOUBLE,
            style="bold cyan",
            expand=False,
            padding=(0, 4),
        ),
        justify="center",
    )
    console.print()

    # Info grid
    console.print(f"  [dim]{'Source':<10}[/dim] {report.source_file}")
    console.print(f"  [dim]{'Duration':<10}[/dim] {_format_time(report.duration_seconds)}")
    console.print(f"  [dim]{'Model':<10}[/dim] {report.whisper_model}")

    flagged_style = "bold red" if report.segments_flagged > 0 else "green"
    console.print(
        f"  [dim]{'Segments':<10}[/dim] {report.segments_total} analyzed, "
        f"[{flagged_style}]{report.segments_flagged} flagged[/{flagged_style}]"
    )
    console.print()

    if report.segments_flagged == 0:
        console.print("  [green]\u2714 No hate speech detected.[/green]")
        console.print()
        return

    # Flagged segments
    flagged = [c for c in report.classifications if c.flagged]
    console.print(
        Rule(
            f"[bold red]Flagged Segments ({len(flagged)})[/bold red]",
            style="red",
        )
    )
    console.print()

    for classification in flagged:
        _print_segment_panel(console, classification)

    # Verbose: clean segments
    if verbose:
        clean = [c for c in report.classifications if not c.flagged]
        if clean:
            console.print(
                Rule(
                    f"[green]Clean Segments ({len(clean)})[/green]",
                    style="green",
                )
            )
            console.print()
            for classification in clean:
                _print_segment_panel(console, classification, dim=True)

    # Summary line
    pct = (report.segments_flagged / report.segments_total * 100) if report.segments_total else 0
    console.print(Rule(style="dim"))
    console.print(
        f"  [dim]{report.segments_flagged} of {report.segments_total} segments flagged "
        f"({pct:.1f}%)[/dim]"
    )
    console.print()


def write_json(report: AnalysisReport, output_path: Path) -> None:
    """Write the full report as JSON."""
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    err = Console(stderr=True, highlight=False)
    err.print(f"  [green]\u2714[/green] JSON report written to: {output_path}")
