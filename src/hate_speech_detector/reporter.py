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


def _build_timeline(report: AnalysisReport, width: int) -> Text:
    """Build a horizontal timeline bar showing flagged segments colored by severity."""
    bar_width = max(20, width - 16)
    duration = report.duration_seconds
    if duration <= 0:
        return Text()

    seconds_per_cell = duration / bar_width
    flagged = [c for c in report.classifications if c.flagged]

    # Time labels
    end_label = _format_time(duration)
    text = Text()
    text.append("00:00", style="dim")
    gap = bar_width - 5 - len(end_label)
    text.append(" " * max(1, gap), style="dim")
    text.append(end_label, style="dim")
    text.append("\n")

    # Bar
    for i in range(bar_width):
        cell_start = i * seconds_per_cell
        cell_end = (i + 1) * seconds_per_cell
        max_score = 0.0
        hit = False
        for c in flagged:
            seg = c.segment
            if seg.start < cell_end and seg.end > cell_start:
                hit = True
                if c.hate_score > max_score:
                    max_score = c.hate_score
        if hit:
            text.append("\u2588", style=_score_style(max_score))
        else:
            text.append("\u2588", style="bright_black")
    text.append("\n")

    # Legend
    text.append("\u2588", style="bold red")
    text.append(" high ", style="dim")
    text.append("\u2588", style="yellow")
    text.append(" mid  ", style="dim")
    text.append("\u2588", style="green")
    text.append(" low  ", style="dim")
    text.append("\u2588", style="bright_black")
    text.append(" clean", style="dim")

    return text


def _build_segment_text(classification: SegmentClassification) -> Text:
    """Build a Rich Text block for a single segment's details."""
    seg = classification.segment
    text = Text()

    # Quoted segment text
    text.append(f'"{seg.text}"', style="bold")

    # LLM reasoning
    if classification.reasoning and classification.flagged:
        text.append("\n\n")
        text.append("LLM reasoning: ", style="bold dim")
        text.append(classification.reasoning, style="italic")
        if classification.embedding_score > 0:
            text.append("\n")
            text.append(f"  embedding pre-filter: {classification.embedding_score:.2f}", style="dim")

    # Context if flagged (show only when no reasoning, to avoid clutter)
    if classification.context and classification.flagged and not classification.reasoning:
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
    if report.language:
        console.print(f"  [dim]{'Language':<10}[/dim] {report.language}")

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

    # Timeline
    if report.duration_seconds > 0:
        timeline = _build_timeline(report, console.width - 6)
        console.print(
            Panel(timeline, title="[dim]Timeline[/dim]", border_style="dim", padding=(0, 1))
        )
        console.print()

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
