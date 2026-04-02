from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console

from hate_speech_detector.extractor import extract_audio
from hate_speech_detector.models import AnalysisReport
from hate_speech_detector.reporter import print_report, write_json

err = Console(stderr=True, highlight=False)


def _is_url(value: str) -> bool:
    """Check if a string looks like a URL."""
    return value.startswith(("http://", "https://"))


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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hate-speech-detector",
        description="Analyze video files for hate speech content.",
    )
    parser.add_argument("video_file", type=str, help="Path to video file or YouTube URL")
    parser.add_argument(
        "--model",
        choices=["tiny", "small", "medium", "large-v3"],
        default="small",
        help="Whisper model size (default: small). Use 'medium' or 'large-v3' for better non-English transcription.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code for transcription (e.g., 'pt', 'en', 'es'). Default: auto-detect.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Hate speech detection threshold 0.0-1.0 (default: 0.20)",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=None,
        help="JSON file with custom reference texts to extend or add categories",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        default=None,
        help="Write full JSON report to file",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Use Claude LLM to verify flagged segments. "
        "Uses `claude` CLI by default (your Claude subscription), "
        "or pass --api-key for direct API access.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key for --verify. If omitted, uses the `claude` CLI instead.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all segments, not just flagged ones",
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cpu"],
        default="mps",
        help="Compute device (default: mps)",
    )

    args = parser.parse_args()

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

    # Step 1: Extract audio
    if is_url:
        from hate_speech_detector.extractor import extract_audio_from_url

        with err.status("  Downloading and extracting audio\u2026"):
            audio_path = extract_audio_from_url(args.video_file)
        err.print("  [green]\u2714[/green] Audio downloaded and extracted")
    else:
        with err.status("  Extracting audio\u2026"):
            audio_path = extract_audio(video_path)
        err.print("  [green]\u2714[/green] Audio extracted")

    # Step 2: Transcribe
    with err.status(f"  Transcribing with Whisper ([bold]{args.model}[/bold])\u2026"):
        from hate_speech_detector.transcriber import transcribe

        segments = transcribe(audio_path, model_name=args.model, language=args.language)
    err.print(f"  [green]\u2714[/green] {len(segments)} segments transcribed")

    # Free Whisper model memory before loading classifiers
    gc.collect()

    # Step 3: Classify
    with err.status("  Loading classification model\u2026"):
        from hate_speech_detector.classifier import HateSpeechClassifier

        classifier = HateSpeechClassifier(
            threshold=args.threshold,
            device=args.device,
            custom_references=custom_references,
        )
    err.print("  [green]\u2714[/green] Classification model loaded")

    with err.status(f"  Classifying {len(segments)} segments\u2026"):
        classifications = classifier.classify(segments)

    flagged_count = sum(1 for c in classifications if c.flagged)
    err.print(f"  [green]\u2714[/green] Embedding pre-filter: [bold]{flagged_count}[/bold] candidates")

    # Step 4: LLM verification (optional)
    if args.verify:
        with err.status("  Verifying with Claude LLM\u2026"):
            from hate_speech_detector.llm_verifier import verify_segments

            classifications = verify_segments(
                classifications,
                api_key=args.api_key,
            )
        flagged_count = sum(1 for c in classifications if c.flagged)
        err.print(f"  [green]\u2714[/green] LLM verified: [bold]{flagged_count}[/bold] flagged")

    # Step 5: Build report
    duration = max((s.end for s in segments), default=0.0)
    flagged_count = sum(1 for c in classifications if c.flagged)

    err.print(f"  [green]\u2714[/green] Analysis complete: [bold]{flagged_count}[/bold] flagged")
    err.print()

    report = AnalysisReport(
        source_file=str(args.video_file),
        duration_seconds=duration,
        whisper_model=args.model,
        segments_total=len(segments),
        segments_flagged=flagged_count,
        classifications=classifications,
    )

    # Step 5: Output
    print_report(report, verbose=args.verbose)

    if args.json_output:
        write_json(report, args.json_output)

    # Cleanup temp audio
    try:
        audio_path.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
