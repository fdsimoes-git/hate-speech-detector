from __future__ import annotations

import argparse
import gc
import shutil
import sys
from pathlib import Path

from rich.console import Console

from hate_speech_detector.extractor import extract_audio
from hate_speech_detector.models import AnalysisReport
from hate_speech_detector.reporter import print_report, write_json

err = Console(stderr=True, highlight=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hate-speech-detector",
        description="Analyze video files for hate speech content.",
    )
    parser.add_argument("video_file", type=Path, help="Path to video file to analyze")
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
        default=0.5,
        help="Hate speech detection threshold 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        default=None,
        help="Write full JSON report to file",
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

    if not args.video_file.exists():
        err.print(f"[bold red]error:[/bold red] file not found: {args.video_file}")
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

    err.print()
    err.print("[bold]hate-speech-detector[/bold]")
    err.print()

    # Step 1: Extract audio
    with err.status("  Extracting audio\u2026"):
        audio_path = extract_audio(args.video_file)
    err.print("  [green]\u2714[/green] Audio extracted")

    # Step 2: Transcribe
    with err.status(f"  Transcribing with Whisper ([bold]{args.model}[/bold])\u2026"):
        from hate_speech_detector.transcriber import transcribe

        segments = transcribe(audio_path, model_name=args.model, language=args.language)
    err.print(f"  [green]\u2714[/green] {len(segments)} segments transcribed")

    # Free Whisper model memory before loading classifiers
    gc.collect()

    # Step 3: Classify
    with err.status("  Loading NLI classification model\u2026"):
        from hate_speech_detector.classifier import HateSpeechClassifier

        classifier = HateSpeechClassifier(
            threshold=args.threshold, device=args.device
        )
    err.print("  [green]\u2714[/green] Classification model loaded")

    with err.status(f"  Classifying {len(segments)} segments\u2026"):
        classifications = classifier.classify(segments)

    # Step 4: Build report
    duration = max((s.end for s in segments), default=0.0)
    flagged_count = sum(1 for c in classifications if c.flagged)

    err.print(f"  [green]\u2714[/green] Classification complete: [bold]{flagged_count}[/bold] flagged")
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
