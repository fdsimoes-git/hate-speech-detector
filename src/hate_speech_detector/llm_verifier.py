from __future__ import annotations

import json
import os
import shutil
import subprocess

from hate_speech_detector.models import (
    CategoryScore,
    SegmentClassification,
)

PREFILTER_THRESHOLD = 0.10

LLM_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are a hate speech analyst. Analyze transcript segments for hate speech content.

Consider:
- Explicit hate speech: slurs, direct attacks on protected groups
- Implied hate speech: dehumanization, stereotyping, coded language, dog whistles
- Cultural and linguistic context: terms or references that carry hateful meaning \
in their specific cultural context
- Surrounding discourse: how neighboring segments inform the meaning

Categories:
- racism: Racial hatred, discrimination, or dehumanization based on race/ethnicity
- sexism: Gender-based hatred, misogyny, or discrimination
- homophobia: Anti-LGBTQ hatred or discrimination
- religious_intolerance: Hatred toward religious groups
- ableism: Discrimination against people with disabilities
- xenophobia: Hatred toward immigrants or foreigners"""


def _build_prompt(candidates: list[tuple[int, SegmentClassification]]) -> str:
    """Build the analysis prompt for a batch of candidate segments."""
    lines = ["Analyze these transcript segments for hate speech.\n"]

    for seg_id, clf in candidates:
        seg = clf.segment
        mm_s, ss_s = divmod(int(seg.start), 60)
        mm_e, ss_e = divmod(int(seg.end), 60)
        lines.append(f"Segment {seg_id} [{mm_s:02d}:{ss_s:02d}\u2013{mm_e:02d}:{ss_e:02d}]:")
        lines.append(f'  Text: "{seg.text}"')
        if clf.context:
            lines.append(f'  Context: "{clf.context}"')
        lines.append("")

    lines.append(
        "For EACH segment, respond with a JSON array:\n"
        "[\n"
        "  {\n"
        '    "segment_id": <number>,\n'
        '    "flagged": <true/false>,\n'
        '    "hate_score": <0.0-1.0>,\n'
        '    "categories": [{"category": "<name>", "score": <0.0-1.0>}],\n'
        '    "reasoning": "<brief explanation>"\n'
        "  }\n"
        "]\n\n"
        "Rules:\n"
        "- Include ALL segments in your response\n"
        "- Only list categories that are relevant (score > 0)\n"
        "- Respond with ONLY the JSON array, no other text"
    )
    return "\n".join(lines)


def _parse_response(text: str) -> list[dict]:
    """Parse the LLM response as a JSON array."""
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def _call_api(prompt: str, model: str, api_key: str) -> str:
    """Call Anthropic API directly with an API key."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_cli(prompt: str, model: str) -> str:
    """Call Claude via the CLI, using the user's Claude subscription."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "claude CLI not found in PATH. Install it or use --api-key instead."
        )

    result = subprocess.run(
        [
            claude_bin,
            "-p",
            "--model", model,
            "--system-prompt", SYSTEM_PROMPT,
            "--output-format", "text",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    return result.stdout


def verify_segments(
    classifications: list[SegmentClassification],
    api_key: str | None = None,
    model: str = LLM_MODEL,
    batch_size: int = 20,
) -> list[SegmentClassification]:
    """Verify pre-filtered segments using Claude LLM.

    Segments with embedding scores >= PREFILTER_THRESHOLD are sent to the LLM
    for reasoning-based verification. The LLM's verdict replaces the embedding
    score for those segments.

    Authentication:
    - If api_key (or ANTHROPIC_API_KEY env var) is set, calls the API directly.
    - Otherwise, uses the `claude` CLI (uses your Claude subscription).
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    use_cli = not key

    # Identify candidates above the pre-filter threshold
    candidates: list[tuple[int, SegmentClassification]] = []
    for i, clf in enumerate(classifications):
        if clf.hate_score >= PREFILTER_THRESHOLD:
            candidates.append((i, clf))

    if not candidates:
        return classifications

    # Process in batches
    verdicts: dict[int, dict] = {}
    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start : batch_start + batch_size]
        prompt = _build_prompt(batch)

        if use_cli:
            raw = _call_cli(prompt, model)
        else:
            raw = _call_api(prompt, model, key)

        parsed = _parse_response(raw)
        for verdict in parsed:
            verdicts[verdict["segment_id"]] = verdict

    # Rebuild classifications with LLM verdicts
    results: list[SegmentClassification] = []
    for i, clf in enumerate(classifications):
        if i in verdicts:
            v = verdicts[i]
            flagged = v.get("flagged", False)
            hate_score = float(v.get("hate_score", 0.0))
            reasoning = v.get("reasoning", "")

            categories: list[CategoryScore] = []
            if flagged:
                for cat_dict in v.get("categories", []):
                    categories.append(
                        CategoryScore(
                            category=cat_dict["category"],
                            score=float(cat_dict["score"]),
                        )
                    )
                categories.sort(key=lambda c: c.score, reverse=True)

            results.append(
                SegmentClassification(
                    segment=clf.segment,
                    hate_score=hate_score,
                    flagged=flagged,
                    categories=categories,
                    context=clf.context,
                    reasoning=reasoning,
                    embedding_score=clf.hate_score,
                )
            )
        else:
            # Below pre-filter threshold — keep as clean
            results.append(
                SegmentClassification(
                    segment=clf.segment,
                    hate_score=clf.hate_score,
                    flagged=False,
                    categories=[],
                    context=clf.context,
                    embedding_score=clf.hate_score,
                )
            )

    return results
