from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console

from hate_speech_detector.pipeline import _is_url
from hate_speech_detector.reporter import print_report, write_json

err = Console(stderr=True, highlight=False)


def _load_custom_references(path: Path) -> dict[str, list[str]]:
    """Load custom reference texts from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        err.print("[bold red]error:[/bold red] references file must be a JSON object")
        sys.exit(1)
    for key, val in data.items():
        if not isinstance(val, list) or not all(isinstance(s, str) for s in val):
            err.print(f"[bold red]error:[/bold red] references['{key}'] must be a list of strings")
            sys.exit(1)
    return data


def _parse_serve_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hate-speech-detector serve",
        description="Start the HTTP API server.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port (default: 8000)",
    )
    parser.add_argument(
        "--device", choices=["mps", "cpu"], default="mps",
        help="Compute device (default: mps)",
    )
    return parser.parse_args(sys.argv[2:])


def _parse_analyze_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hate-speech-detector",
        description="Analyze video files for hate speech content.",
    )
    parser.add_argument("video_file", type=str, help="Path to video file or YouTube URL")
    parser.add_argument(
        "--model",
        choices=["tiny", "small", "medium", "large-v3"],
        default="small",
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Language code for transcription (e.g., 'pt', 'en', 'es'). Default: auto-detect.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.20,
        help="Hate speech detection threshold 0.0-1.0 (default: 0.20)",
    )
    parser.add_argument(
        "--references", type=Path, default=None,
        help="JSON file with custom reference texts to extend or add categories",
    )
    parser.add_argument(
        "--json", dest="json_output", type=Path, default=None,
        help="Write full JSON report to file",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Use Claude LLM to verify flagged segments.",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Anthropic API key for --verify. If omitted, uses the `claude` CLI instead.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show all segments, not just flagged ones",
    )
    parser.add_argument(
        "--device", choices=["mps", "cpu"], default="mps",
        help="Compute device (default: mps)",
    )
    return parser.parse_args()


def main() -> None:
    # Check if first arg is the "serve" subcommand
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        args = _parse_serve_args()
        _run_serve(args)
        return

    args = _parse_analyze_args()
    _run_analyze(args)


def _run_analyze(args: argparse.Namespace) -> None:
    """Run the CLI analysis pipeline."""
    is_url = _is_url(args.video_file)

    if is_url:
        if not shutil.which("yt-dlp"):
            err.print(
                "[bold red]error:[/bold red] yt-dlp is required for URL input but not found.\n"
                "       Install it with: [bold]brew install yt-dlp[/bold]"
            )
            sys.exit(1)
    else:
        video_path = Path(args.video_file)
        if not video_path.exists():
            err.print(f"[bold red]error:[/bold red] file not found: {video_path}")
            sys.exit(1)

    if not 0.0 <= args.threshold <= 1.0:
        err.print("[bold red]error:[/bold red] threshold must be between 0.0 and 1.0")
        sys.exit(1)

    if not shutil.which("ffmpeg"):
        err.print(
            "[bold red]error:[/bold red] ffmpeg is required but not found.\n"
            "       Install it with: [bold]brew install ffmpeg[/bold]"
        )
        sys.exit(1)

    custom_references = None
    if args.references:
        if not args.references.exists():
            err.print(f"[bold red]error:[/bold red] references file not found: {args.references}")
            sys.exit(1)
        custom_references = _load_custom_references(args.references)

    err.print()
    err.print("[bold]hate-speech-detector[/bold]")
    err.print()

    from hate_speech_detector.pipeline import analyze

    with err.status("  Analyzing\u2026"):
        report = analyze(
            source=args.video_file,
            model=args.model,
            language=args.language,
            threshold=args.threshold,
            device=args.device,
            verify=args.verify,
            api_key=args.api_key,
            custom_references=custom_references,
        )

    err.print(f"  [green]\u2714[/green] Analysis complete: [bold]{report.segments_flagged}[/bold] flagged")
    err.print()

    print_report(report, verbose=args.verbose)

    if args.json_output:
        write_json(report, args.json_output)


def _run_serve(args: argparse.Namespace) -> None:
    """Start the HTTP API server."""
    try:
        import uvicorn
    except ImportError:
        err.print(
            "[bold red]error:[/bold red] server dependencies not installed.\n"
            "       Install with: [bold]uv sync --group server[/bold]"
        )
        sys.exit(1)

    err.print()
    err.print("[bold]hate-speech-detector server[/bold]")
    err.print(f"  Listening on [bold]http://{args.host}:{args.port}[/bold]")
    err.print(f"  API docs at  [bold]http://{args.host}:{args.port}/docs[/bold]")
    err.print()

    import os
    os.environ["HSD_DEVICE"] = args.device

    uvicorn.run(
        "hate_speech_detector.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
