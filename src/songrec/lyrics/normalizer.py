from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s'’-]", flags=re.UNICODE)


def normalize_lyrics_text(text: str) -> str:
    """
    Normalize ASR output for lyrics search.

    This keeps words, digits, apostrophes and dashes, but removes most punctuation
    and collapses whitespace. It is intentionally language-agnostic, so it works
    for English, Russian and mixed lyrics.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = normalized.replace("’", "'")
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()
