"""Unit tests for jarvis.memory.chunker."""

from __future__ import annotations

import logging

import pytest

from jarvis.memory import chunker
from jarvis.memory.chunker import (
    Chunk,
    chunk_markdown,
    configure_tokenizer,
    count_tokens,
)


@pytest.fixture(autouse=True)
def _reset_tokenizer():
    """Each test starts with a clean tokenizer state set to approximation."""
    configure_tokenizer("approximation")
    yield
    configure_tokenizer("approximation")


def test_count_tokens_falls_back_to_approximation(monkeypatch):
    # Force the qwen-native loader to fail.
    monkeypatch.setattr(chunker, "_try_load_qwen_tokenizer", lambda: None)
    resolved = configure_tokenizer("qwen-native")
    assert resolved == "approximation"
    # Sanity: 4-char rule.
    assert count_tokens("abcd") == 1
    assert count_tokens("a" * 16) == 4
    assert count_tokens("") == 0


def test_qwen_unavailable_emits_error_log(monkeypatch, caplog):
    monkeypatch.setattr(chunker, "_try_load_qwen_tokenizer", lambda: None)
    with caplog.at_level(logging.ERROR, logger="jarvis.memory.chunker"):
        configure_tokenizer("qwen-native")
    assert any(
        "qwen-native" in r.message and r.levelno == logging.ERROR for r in caplog.records
    ), caplog.records


def test_chunk_empty_file_returns_empty_list():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n   \n") == []


def test_chunk_frontmatter_only_returns_empty_list():
    text = "---\nkey: value\nother: 1\n---\n"
    assert chunk_markdown(text) == []


def test_chunk_single_short_section_one_chunk():
    text = "# Heading\n\nSome short body text.\n"
    out = chunk_markdown(text)
    assert len(out) == 1
    c = out[0]
    assert c.heading_path == "# Heading"
    assert c.start_line == 1
    assert c.end_line == 3  # heading + blank + body
    assert "Some short body text." in c.content


def test_chunk_no_headings_pre_heading_bucket():
    text = "Just a body with no headings at all.\nLine two.\n"
    out = chunk_markdown(text)
    assert len(out) == 1
    assert out[0].heading_path == ""
    assert out[0].start_line == 1
    assert out[0].end_line == 2


def test_heading_path_nested():
    text = (
        "## Projects\n\n"
        "Top-level body.\n\n"
        "### rocket-sim\n\n"
        "Nested body content here.\n"
    )
    out = chunk_markdown(text)
    # Two sections.
    assert len(out) == 2
    paths = [c.heading_path for c in out]
    assert "## Projects" in paths
    assert "## Projects > ### rocket-sim" in paths


def test_chunk_line_numbers_account_for_frontmatter():
    text = "---\nkey: v\n---\n# H\nbody\n"
    out = chunk_markdown(text)
    assert len(out) == 1
    c = out[0]
    # Frontmatter consumes 3 lines ("---", "key: v", "---"). "# H" is line 4 in the raw file.
    assert c.start_line == 4
    assert c.end_line == 5


def test_chunk_huge_section_sliding_window(monkeypatch):
    # Stub tokenizer to one token per word so we can hit the threshold deterministically.
    monkeypatch.setattr(chunker, "count_tokens", lambda text: max(1, len(text.split())))

    paragraphs = []
    for i in range(20):
        paragraphs.append(" ".join([f"para{i}word{j}" for j in range(50)]))  # 50 tokens each
    body = "\n\n".join(paragraphs)
    text = f"# Big\n\n{body}\n"

    out = chunk_markdown(text, target_tokens=200, overlap_tokens=50)
    assert len(out) >= 2

    # Line ranges cover the section (the heading line is 1; body starts line 3).
    starts = [c.start_line for c in out]
    ends = [c.end_line for c in out]
    assert min(starts) <= 3
    assert max(ends) >= 3
    # Successive chunks should overlap (next.start_line <= prev.end_line for some pair).
    overlap_seen = any(out[i + 1].start_line <= out[i].end_line for i in range(len(out) - 1))
    assert overlap_seen, "expected at least one overlapping window"


def test_chunk_returns_chunk_dataclass():
    text = "# H\n\nbody\n"
    out = chunk_markdown(text)
    assert all(isinstance(c, Chunk) for c in out)
    assert all(c.token_count > 0 for c in out)
