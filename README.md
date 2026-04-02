# hate-speech-detector

A command-line tool that analyzes video files for hate speech content. It extracts audio, transcribes it using Whisper, and uses a hybrid approach — multilingual embeddings for fast pre-filtering, then Claude LLM for reasoning-based verification — to detect racism, sexism, homophobia, religious intolerance, ableism, and xenophobia with timestamped scores.

## How it works

```
Video file or YouTube URL
    |
    v
[yt-dlp] ──> (if URL) Download audio-only stream
    |
    v
[ffmpeg] ──> Extract/convert audio (16kHz mono WAV)
    |
    v
[Whisper] ──> Transcribe speech to timestamped segments
    |
    v
[Embeddings] ──> Pre-filter: cosine similarity against hate speech references
    |
    v (optional, --verify)
[Claude LLM] ──> Verify: reasoning-based analysis of flagged candidates
    |
    v
Formatted report with timeline, scores, categories, and reasoning
```

1. **Input** — Accepts a local video file or a YouTube URL. For URLs, yt-dlp downloads only the audio stream (no video stored on disk).
2. **Audio extraction** — ffmpeg converts the audio to a 16kHz mono WAV file
3. **Transcription** — Whisper (via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on Apple Silicon) transcribes speech into timestamped text segments
4. **Embedding pre-filter** — Each segment (with context from neighboring segments) is encoded using a multilingual sentence-transformer and scored via cosine similarity against multiple reference texts per hate speech category. Segments above the pre-filter threshold (0.10) become candidates.
5. **LLM verification** (with `--verify`) — Candidate segments are sent to Claude for reasoning-based analysis. The LLM evaluates cultural context, coded language, dog whistles, and implied meaning — catching what embeddings miss and eliminating false positives.
6. **Reporting** — Results include a color-coded timeline showing where hate speech occurs, detailed panels with score bars and LLM reasoning, and optional JSON export.

### Why hybrid instead of embeddings alone?

Embeddings measure **surface similarity** — they can tell that a text is *near* hate speech references, but they can't reason about context. For example, "o afrodescendente mais leve pesava sete arrobas" (comparing Black people to cattle using livestock weight units) requires cultural knowledge to recognize as dehumanizing. The LLM verification step provides this reasoning capability while embeddings keep the process fast by filtering out clearly irrelevant segments.

### Context window

Short segments like *"They don't do anything"* are harmless in isolation but hateful when preceded by *"Those indigenous communities..."*. Each segment is scored with its neighboring segments concatenated, so both the embedding model and the LLM see the full conversational context.

## Requirements

- **Python** >= 3.11
- **ffmpeg** — `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)
- **yt-dlp** (optional) — `brew install yt-dlp` — required only for analyzing YouTube URLs
- ~2GB disk space for model downloads (cached after first run)
- **Claude CLI** or **Anthropic API key** (only for `--verify` mode) — by default uses the `claude` CLI (your Claude subscription). Or pass `--api-key` / set `ANTHROPIC_API_KEY` for direct API access.

### Hardware

Runs on CPU or Apple Silicon GPU (MPS). Tested on MacBook Pro M3 with 8GB RAM. Models load sequentially with memory cleanup between stages, so 8GB is sufficient.

## Installation

```bash
# Clone the repository
git clone https://github.com/fdsimoes-git/hate-speech-detector.git
cd hate-speech-detector

# Install with uv (recommended)
uv sync

# Or with pip
pip install .
```

## Usage

```bash
# Basic usage (embedding-only, no API key needed)
hate-speech-detector video.mp4

# Analyze a YouTube video directly (no video stored on disk)
hate-speech-detector "https://www.youtube.com/watch?v=VIDEO_ID" --language pt --verify

# With LLM verification for higher accuracy
hate-speech-detector video.mp4 --verify

# Specify language for better transcription accuracy
hate-speech-detector video.mp4 --language pt --verify

# Use a larger Whisper model for non-English content
hate-speech-detector video.mp4 --language pt --model large-v3 --verify

# Lower threshold to catch more subtle cases
hate-speech-detector video.mp4 --threshold 0.3

# Show all segments, not just flagged ones
hate-speech-detector video.mp4 --verbose --verify

# Export full report as JSON
hate-speech-detector video.mp4 --json report.json --verify

# With LLM verification — uses `claude` CLI (your Claude subscription, no API key needed)
hate-speech-detector video.mp4 --verify

# Or use a direct API key instead
hate-speech-detector video.mp4 --verify --api-key sk-ant-...

# Add custom reference texts to improve detection
hate-speech-detector video.mp4 --references custom_refs.json --verify

# Force CPU if MPS causes issues
hate-speech-detector video.mp4 --device cpu
```

### Options

| Option | Description | Default |
|---|---|---|
| `video_file` | Path to video file or YouTube URL | (required) |
| `--model` | Whisper model size: `tiny`, `small`, `medium`, `large-v3` | `small` |
| `--language` | Language code (e.g., `pt`, `en`, `es`). Auto-detects if omitted | auto |
| `--threshold` | Detection threshold 0.0–1.0. Lower = more sensitive | `0.20` |
| `--verify` | Enable Claude LLM verification of flagged segments | off |
| `--api-key` | Anthropic API key for `--verify`. If omitted, uses `claude` CLI instead | — |
| `--references` | JSON file with custom reference texts to extend categories | — |
| `--json PATH` | Write full JSON report to file | — |
| `--verbose` | Show all segments, not just flagged ones | off |
| `--device` | Compute device: `mps` (Apple Silicon) or `cpu` | `mps` |

### Choosing a threshold

- **0.20** (default) — balanced sensitivity for the embedding pre-filter
- **0.10–0.15** — more sensitive pre-filter, sends more candidates to LLM verification
- **0.30+** — less sensitive, only obvious matches. Good for embedding-only mode without `--verify`

### Custom references

You can extend or add categories with a JSON file:

```json
{
  "racism": [
    "using weight units for livestock to describe people of a certain race",
    "referring to Black people using cattle or farm animal terminology"
  ],
  "political_extremism": [
    "calls for political violence, armed uprising, or overthrowing a government"
  ]
}
```

Custom references are merged with the built-in ones. New category names (like `political_extremism`) create new detection categories.

## Example output

```
hate-speech-detector

  ✔ Audio extracted
  ✔ 13 segments transcribed
  ✔ Classification model loaded
  ✔ Embedding pre-filter: 13 candidates
  ✔ LLM verified: 5 flagged
  ✔ Analysis complete: 5 flagged

                  ╔══════════════════════════════════════════╗
                  ║   Hate Speech Analysis Report            ║
                  ╚══════════════════════════════════════════╝

  Source     video.mp4
  Duration   00:00:53
  Model      large-v3
  Segments   13 analyzed, 5 flagged

╭─ Timeline ───────────────────────────────────────────────╮
│ 00:00                                         00:00:53   │
│ ████████████████████░░░░░░████████████████████████████    │
│ █ high  █ mid  █ low  █ clean                            │
╰──────────────────────────────────────────────────────────╯

──── Flagged Segments (5) ──────────────────────────────────

╭─ 00:00:33 → 00:00:41 ──────────────── score: 0.92 ──────╮
│                                                           │
│  "Olha, o afrodescendente mais leve lá pesava sete       │
│   arrobas."                                               │
│                                                           │
│  LLM reasoning: Uses "arrobas" (a unit for weighing      │
│  cattle) to describe Black people, dehumanizing them      │
│  by equating them with livestock.                         │
│    embedding pre-filter: 0.39                             │
│                                                           │
│  racism               0.92 ██████████████████░░          │
│  ableism              0.15 ███░░░░░░░░░░░░░░░░░          │
│                                                           │
╰───────────────────────────────────── score: 0.92 ────────╯
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
| Embedding pre-filter | [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) | 278M params | Multilingual sentence embeddings |
| LLM verification | [Claude Haiku](https://www.anthropic.com) | — | Reasoning-based hate speech analysis |

Whisper and the sentence-transformer are downloaded from Hugging Face on first run and cached locally. Claude Haiku requires an API key.

## Server Mode

Run as an HTTP API server so other machines on your LAN can submit analysis requests:

```bash
# Install server dependencies
uv sync --group server

# Start the server (binds to all interfaces by default)
hate-speech-detector serve

# Custom host/port
hate-speech-detector serve --host 0.0.0.0 --port 9000

# Force CPU
hate-speech-detector serve --device cpu
```

The server exposes:
- `GET /health` — health check
- `POST /analyze` — analyze a video (file upload or URL)
- `GET /docs` — interactive API documentation (Swagger UI)

### API usage examples

```bash
# Analyze a YouTube URL
curl -X POST http://192.168.1.x:8000/analyze \
  -F "url=https://www.youtube.com/watch?v=VIDEO_ID" \
  -F "language=pt" \
  -F "model=large-v3" \
  -F "verify=true"

# Upload a video file
curl -X POST http://192.168.1.x:8000/analyze \
  -F "file=@video.mp4" \
  -F "language=en" \
  -F "verify=true"
```

The response is the same JSON structure as `--json` output.

### Server options

| Option | Description | Default |
|---|---|---|
| `--host` | Bind address | `0.0.0.0` |
| `--port` | Port | `8000` |
| `--device` | Compute device: `mps` or `cpu` | `mps` |

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_llm_verifier.py -v
```

## Architecture

```
src/hate_speech_detector/
├── cli.py           # Entry point, argument parsing, serve command
├── pipeline.py      # Core analysis pipeline (shared by CLI and server)
├── server.py        # FastAPI HTTP server for LAN access
├── extractor.py     # Video/URL → audio extraction via ffmpeg/yt-dlp
├── transcriber.py   # Audio → timestamped text segments via Whisper
├── classifier.py    # Text → embedding scores via sentence-transformers
├── llm_verifier.py  # Embedding candidates → LLM-verified verdicts via Claude
├── reporter.py      # Scores → timeline + formatted terminal/JSON reports
└── models.py        # Data classes (TranscriptSegment, CategoryScore, etc.)
```

## License

[MIT](LICENSE)
