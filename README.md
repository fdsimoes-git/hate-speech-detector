# hate-speech-detector

A command-line tool that analyzes video files for hate speech content. It extracts audio, transcribes it using Whisper, and classifies each segment using a multilingual zero-shot NLI (Natural Language Inference) model to detect racism, sexism, homophobia, religious intolerance, ableism, and xenophobia — with timestamped probability scores.

## How it works

```
Video file
    |
    v
[ffmpeg] ──> Extract audio (16kHz mono WAV)
    |
    v
[Whisper] ──> Transcribe speech to timestamped segments
    |
    v
[mDeBERTa NLI] ──> Classify each segment against hate speech categories
    |
    v
Formatted report with scores per category
```

1. **Audio extraction** — ffmpeg converts the video to a 16kHz mono WAV file
2. **Transcription** — Whisper (via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on Apple Silicon) transcribes speech into timestamped text segments
3. **Classification** — Each segment (with surrounding context from neighboring segments) is evaluated by a zero-shot NLI classifier ([mDeBERTa-v3-base-xnli-multilingual-nli-2mil7](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)) against six hate speech categories
4. **Reporting** — Results are displayed as a color-coded terminal report with score bars, and optionally exported as JSON

### Why NLI instead of keyword matching or embeddings?

The NLI model evaluates whether a text **entails** a hypothesis like *"This text contains racist speech."* This means it catches **implied** hate speech — not just slurs, but dehumanizing language, dog whistles, and coded prejudice — across any language the model supports.

### Context window

Short segments like *"They don't do anything"* are harmless in isolation but hateful when preceded by *"Those indigenous communities..."*. Each segment is scored with its neighboring segments concatenated, so the model sees the full conversational context. The report shows both the individual segment and the context that was scored.

## Requirements

- **Python** >= 3.11
- **ffmpeg** — `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)
- ~2GB disk space for model downloads (cached after first run)

### Hardware

Runs on CPU or Apple Silicon GPU (MPS). Tested on MacBook Pro M3 with 8GB RAM. Models load sequentially with memory cleanup between stages, so 8GB is sufficient.

## Installation

```bash
# Clone the repository
git clone https://github.com/felipesimoes/hate-speech-detector.git
cd hate-speech-detector

# Install with uv (recommended)
uv sync

# Or with pip
pip install .
```

## Usage

```bash
# Basic usage
hate-speech-detector video.mp4

# Specify language for better transcription accuracy
hate-speech-detector video.mp4 --language pt

# Use a larger Whisper model for non-English content
hate-speech-detector video.mp4 --language pt --model large-v3

# Lower threshold to catch more subtle cases
hate-speech-detector video.mp4 --threshold 0.3

# Show all segments, not just flagged ones
hate-speech-detector video.mp4 --verbose

# Export full report as JSON
hate-speech-detector video.mp4 --json report.json

# Force CPU if MPS causes issues
hate-speech-detector video.mp4 --device cpu
```

### Options

| Option | Description | Default |
|---|---|---|
| `video_file` | Path to video file to analyze | (required) |
| `--model` | Whisper model size: `tiny`, `small`, `medium`, `large-v3` | `small` |
| `--language` | Language code (e.g., `pt`, `en`, `es`). Auto-detects if omitted | auto |
| `--threshold` | Detection threshold 0.0–1.0. Lower = more sensitive | `0.5` |
| `--json PATH` | Write full JSON report to file | — |
| `--verbose` | Show all segments, not just flagged ones | off |
| `--device` | Compute device: `mps` (Apple Silicon) or `cpu` | `mps` |

### Choosing a threshold

- **0.5** (default) — flags clear, unambiguous hate speech
- **0.3–0.4** — catches more subtle or implied cases, with some false positives
- **0.2** — very sensitive, useful for screening

## Example output

```
hate-speech-detector

  ✓ Audio extracted
  ✓ 45 segments transcribed
  ✓ Classification model loaded
  ✓ Classification complete: 3 flagged

                  ╔══════════════════════════════════════════╗
                  ║   Hate Speech Analysis Report            ║
                  ╚══════════════════════════════════════════╝

  Source     video.mp4
  Duration   00:12:34
  Model      large-v3
  Segments   45 analyzed, 3 flagged

──── Flagged Segments (3) ──────────────────────────────────

╭─ 00:00:42 → 00:00:44 ──────────────── score: 0.70 ──────╮
│                                                           │
│  "Não fazem nada."                                       │
│                                                           │
│  Scored with context:                                     │
│  "O afrodescendente mais leve pesava sete arrobas.       │
│  Não fazem nada. Vivem de bolsa do governo."             │
│                                                           │
│  racism               0.70 ██████████████░░░░░░          │
│  xenophobia           0.36 ███████░░░░░░░░░░░░░          │
│  sexism               0.12 ██░░░░░░░░░░░░░░░░░░          │
│                                                           │
╰───────────────────────────────────── score: 0.70 ────────╯
```

## Categories detected

| Category | What it catches |
|---|---|
| **racism** | Racial discrimination, dehumanization based on race/ethnicity |
| **sexism** | Misogyny, gender-based discrimination |
| **homophobia** | Anti-LGBTQ discrimination |
| **religious_intolerance** | Hatred toward religious groups |
| **ableism** | Discrimination against people with disabilities |
| **xenophobia** | Hatred toward immigrants and foreigners |

## Models used

| Stage | Model | Size | Purpose |
|---|---|---|---|
| Transcription | [Whisper](https://github.com/openai/whisper) (via mlx-whisper) | 39M–1.5B params | Speech-to-text with timestamps |
| Classification | [mDeBERTa-v3-base-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) | 279M params | Multilingual zero-shot NLI |

Both models are downloaded from Hugging Face on first run and cached locally.

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_classifier.py -v
```

## Architecture

```
src/hate_speech_detector/
├── cli.py           # Entry point, argument parsing, pipeline orchestration
├── extractor.py     # Video → audio extraction via ffmpeg
├── transcriber.py   # Audio → timestamped text segments via Whisper
├── classifier.py    # Text → hate speech scores via zero-shot NLI
├── reporter.py      # Scores → formatted terminal/JSON reports
└── models.py        # Data classes (TranscriptSegment, CategoryScore, etc.)
```

## License

[MIT](LICENSE)
