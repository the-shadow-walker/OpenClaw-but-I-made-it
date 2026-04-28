"""Heading-aware Markdown chunker (BUILD_SPEC §6.2).

Output ``Chunk`` records carry 1-indexed inclusive line ranges that point at the
*raw* file (frontmatter included). That precision is what lets the future
``memory_get`` tool quote a chunk verbatim from the on-disk Markdown.

Frontmatter handling — explicit regex, NOT python-frontmatter
-------------------------------------------------------------
``python-frontmatter.loads()`` returns ``post.content`` but does NOT report how
many source lines the frontmatter consumed. That makes line-number anchoring
brittle. We strip frontmatter with an explicit regex so we can compute and add
back a ``frontmatter_line_offset`` to every emitted chunk's start_line/end_line.
Do not "fix" this back to python-frontmatter — it will silently break
line-number anchoring and the symptom will only show up when memory_get returns
the wrong text. (Sticky comment for the next maintainer.)

Tokenizer (§17.1)
-----------------
Two implementations:
  * ``approximation`` — ``max(1, len(text) // 4)``. No external deps. Default
    fallback when transformers is unavailable.
  * ``qwen-native`` — ``transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")``,
    lazily loaded as a module-level singleton.

``configure_tokenizer(kind)`` is called once at CLI startup. It resolves the
choice (falling back to approximation if qwen-native was requested but
transformers is not installed) and returns the kind that won so the caller
can log it.

Loudness rules:
  * Requested approximation → resolves silently to approximation. INFO log
    happens at the CLI layer ("using tokenizer: approximation").
  * Requested qwen-native AND transformers available → INFO log at CLI.
  * Requested qwen-native AND transformers NOT available → ERROR log here
    (single emission, never raise) so a fresh deploy that lost the
    ``[qwen-tokenizer]`` extra is loudly visible. The ±15% chunk-size drift
    of the approximation compounds into wrong P6+ compaction triggers, so
    silent fallback is unacceptable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

__all__ = [
    "Chunk",
    "configure_tokenizer",
    "count_tokens",
    "chunk_markdown",
]


# ---------------------------------------------------------------------------
# Tokenizer state — module-level singletons resolved by configure_tokenizer().
# ---------------------------------------------------------------------------

# What was actually resolved (after fallback). None until configure_tokenizer runs.
_RESOLVED_KIND: Literal["qwen-native", "approximation"] | None = None
_QWEN_TOKENIZER = None  # opaque: a transformers tokenizer or None

_QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B"  # §19 anti-pattern #10 exception: this is a tokenizer constant.


def _try_load_qwen_tokenizer():
    """Best-effort load of the Qwen tokenizer. Returns the tokenizer or None."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return AutoTokenizer.from_pretrained(_QWEN_MODEL_NAME)
    except Exception:
        # Network unavailable, model not cached, etc. Fall back rather than crash.
        return None


def configure_tokenizer(kind: Literal["qwen-native", "approximation"]) -> Literal["qwen-native", "approximation"]:
    """Resolve the tokenizer choice. Returns the kind that won (post-fallback).

    Idempotent: calling repeatedly with the same kind is a no-op after the
    first call. Calling with a different kind re-resolves.
    """
    global _RESOLVED_KIND, _QWEN_TOKENIZER

    if kind == "approximation":
        _RESOLVED_KIND = "approximation"
        _QWEN_TOKENIZER = None
        return "approximation"

    if kind == "qwen-native":
        tok = _try_load_qwen_tokenizer()
        if tok is None:
            logger.error(
                "requested tokenizer 'qwen-native' but transformers is not installed; "
                "falling back to approximation. Chunk-size estimates will drift ±15%, "
                "which compounds into wrong compaction triggers in P6+. "
                "Install with: pip install jarvis[qwen-tokenizer]"
            )
            _RESOLVED_KIND = "approximation"
            _QWEN_TOKENIZER = None
            return "approximation"
        _RESOLVED_KIND = "qwen-native"
        _QWEN_TOKENIZER = tok
        return "qwen-native"

    raise ValueError(f"unknown tokenizer kind: {kind!r}")


def count_tokens(text: str) -> int:
    """Return token count for ``text`` using the configured tokenizer.

    If ``configure_tokenizer`` has not been called yet, defaults to the
    approximation. Always returns at least 1 for non-empty inputs (matches
    the approximation contract).
    """
    if not text:
        return 0
    if _RESOLVED_KIND == "qwen-native" and _QWEN_TOKENIZER is not None:
        # `encode` returns a list of ids; `add_special_tokens=False` matches
        # how chunked content actually flows into the model body.
        return max(1, len(_QWEN_TOKENIZER.encode(text, add_special_tokens=False)))
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    content: str
    start_line: int       # 1-indexed inclusive (in the raw file, frontmatter included)
    end_line: int         # 1-indexed inclusive
    heading_path: str     # e.g. "## Projects > ### rocket-sim", or "" for pre-heading content
    token_count: int


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


# Frontmatter detector. Matches a leading "---\n...\n---\n" block.
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _strip_frontmatter(text: str) -> tuple[str, int]:
    """Return (body_text, lines_consumed_by_frontmatter)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return text, 0
    consumed = m.group(0).count("\n")
    return text[m.end():], consumed


def _split_by_heading(text: str) -> list[tuple[str, str, int, int]]:
    """Split text into sections by ATX headings.

    Returns a list of ``(heading_path, body, line_start, line_end)`` tuples
    where line numbers are 1-indexed inclusive *relative to the input text*
    (the caller adds the frontmatter offset).

    A "section" is the lines between one heading and the next (inclusive of
    the heading line itself). Pre-heading content (before any ``#`` line) is
    emitted as a section with ``heading_path=""``.
    """
    lines = text.splitlines()
    if not lines:
        return []

    sections: list[tuple[str, str, int, int]] = []
    stack: list[tuple[int, str]] = []  # (level, "## Title") for path building

    cur_heading_path = ""
    cur_start_line = 1
    cur_lines: list[str] = []

    def flush(end_line: int) -> None:
        if not cur_lines:
            return
        body = "\n".join(cur_lines)
        # Skip whitespace-only sections.
        if not body.strip():
            return
        sections.append((cur_heading_path, body, cur_start_line, end_line))

    for i, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if m:
            # Close out the previous section (ends on the line BEFORE this heading).
            flush(i - 1)
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_token = f"{'#' * level} {title}"

            # Pop deeper-or-equal levels off the stack so we sit at our parent.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_token))
            cur_heading_path = " > ".join(tok for _, tok in stack)
            cur_start_line = i
            cur_lines = [line]
        else:
            cur_lines.append(line)

    # Final flush — last section runs to EOF.
    flush(len(lines))
    return sections


def _split_paragraphs(body_lines: list[str], body_start_line: int) -> list[tuple[str, int, int]]:
    """Split body lines into paragraphs (blank-line separated).

    Returns ``[(text, start_line, end_line), ...]`` with absolute line numbers
    (computed from ``body_start_line``).
    """
    paragraphs: list[tuple[str, int, int]] = []
    cur: list[str] = []
    cur_start: int | None = None

    for offset, line in enumerate(body_lines):
        absolute_line = body_start_line + offset
        if line.strip() == "":
            if cur:
                end_line = cur_start + len(cur) - 1  # type: ignore[operator]
                paragraphs.append(("\n".join(cur), cur_start, end_line))  # type: ignore[arg-type]
                cur = []
                cur_start = None
            continue
        if cur_start is None:
            cur_start = absolute_line
        cur.append(line)

    if cur:
        end_line = cur_start + len(cur) - 1  # type: ignore[operator]
        paragraphs.append(("\n".join(cur), cur_start, end_line))  # type: ignore[arg-type]

    return paragraphs


def _sliding_window(
    body: str,
    body_start_line: int,
    heading_path: str,
    target_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Greedy paragraph-packing with tail overlap.

    Fills windows with whole paragraphs up to ``target_tokens``. Then, for
    overlap, prepends the tail-end paragraphs of the previous window summing
    to ~``overlap_tokens`` to the next window. Each window's ``start_line`` /
    ``end_line`` is the union of contributing paragraph ranges.
    """
    body_lines = body.splitlines()
    paragraphs = _split_paragraphs(body_lines, body_start_line)
    if not paragraphs:
        return []

    # Pre-tokenize each paragraph once.
    para_tokens = [count_tokens(p) for p, _, _ in paragraphs]

    chunks: list[Chunk] = []
    i = 0
    n = len(paragraphs)
    while i < n:
        window: list[int] = []  # indices into paragraphs
        window_tokens = 0

        # If we have prior chunks, prepend tail-overlap paragraphs from the
        # previous window: walk backwards from i-1 collecting paragraphs whose
        # cumulative tokens hit overlap_tokens. (Skip when starting fresh.)
        if i > 0 and overlap_tokens > 0:
            overlap_indices: list[int] = []
            overlap_sum = 0
            j = i - 1
            while j >= 0 and overlap_sum < overlap_tokens:
                overlap_indices.insert(0, j)
                overlap_sum += para_tokens[j]
                j -= 1
            window.extend(overlap_indices)
            window_tokens += overlap_sum

        # Greedily pack the next paragraphs until we'd exceed target_tokens.
        # If a single paragraph alone exceeds target_tokens, we still include
        # it on its own (one-paragraph chunk) — better than dropping content.
        progress_made = False
        while i < n:
            tk = para_tokens[i]
            if window_tokens + tk > target_tokens and progress_made:
                break
            window.append(i)
            window_tokens += tk
            i += 1
            progress_made = True

        if not window:
            break

        # Materialize the chunk.
        contents = [paragraphs[k][0] for k in window]
        start_line = min(paragraphs[k][1] for k in window)
        end_line = max(paragraphs[k][2] for k in window)
        chunk_text = "\n\n".join(contents)
        chunks.append(
            Chunk(
                content=chunk_text,
                start_line=start_line,
                end_line=end_line,
                heading_path=heading_path,
                token_count=count_tokens(chunk_text),
            )
        )

    return chunks


def chunk_markdown(text: str, target_tokens: int = 400, overlap_tokens: int = 80) -> list[Chunk]:
    """Chunk ``text`` into heading-aware ``Chunk`` records.

    Algorithm:
      1. Strip leading YAML frontmatter via regex; remember how many lines
         were consumed so we can offset emitted chunks back to raw-file lines.
      2. Walk the post-frontmatter body line-by-line, splitting on ATX
         headings. Each section becomes one or more chunks.
      3. If a section's body fits in ``target_tokens``, emit a single chunk.
         Otherwise, sliding-window the body by paragraphs with overlap.

    Empty input, frontmatter-only input, and whitespace-only sections all
    return ``[]`` (or skip cleanly). No exceptions on degenerate input.
    """
    if not text or not text.strip():
        return []

    body, fm_offset = _strip_frontmatter(text)
    if not body.strip():
        return []

    sections = _split_by_heading(body)
    chunks: list[Chunk] = []

    for heading_path, section_body, line_start, line_end in sections:
        section_tokens = count_tokens(section_body)
        if section_tokens <= target_tokens:
            chunks.append(
                Chunk(
                    content=section_body,
                    start_line=line_start + fm_offset,
                    end_line=line_end + fm_offset,
                    heading_path=heading_path,
                    token_count=section_tokens,
                )
            )
        else:
            sub = _sliding_window(
                section_body,
                body_start_line=line_start,
                heading_path=heading_path,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
            # Apply the frontmatter offset to each emitted chunk.
            for c in sub:
                chunks.append(
                    Chunk(
                        content=c.content,
                        start_line=c.start_line + fm_offset,
                        end_line=c.end_line + fm_offset,
                        heading_path=c.heading_path,
                        token_count=c.token_count,
                    )
                )

    return chunks
