"""Text cleaning and the public-title sanitiser (design 9.3 and 11)."""

from __future__ import annotations

import re
import unicodedata

PUBLIC_TITLE_MAX = 80
LINK_REMOVED = "[link removed]"
UNTITLED = "Untitled"

_DROP_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}
_KEEP_CONTROLS = {"\n", "\t"}

_SCHEME_URL = re.compile(r"\b[a-z][a-z0-9+.\-]*://\S+", re.IGNORECASE)
_SCRIPT_URL = re.compile(r"\b(?:javascript|vbscript|data|file|mailto|tel):\S+", re.IGNORECASE)
_WWW = re.compile(r"\bwww\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+", re.UNICODE)
# Bare host.tld or host.tld/path: last label 2-24 letters, so "v1.2" and "0.5px" survive.
_BARE_HOST = re.compile(
    r"(?<![\w@])(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?![\w\-])(?:[/:?#]\S*)?",
    re.IGNORECASE,
)
_SPACES = re.compile(r"\s+")


def _drop_invisible(text: str, keep_newlines: bool) -> str:
    out = []
    for ch in text:
        if keep_newlines and ch in _KEEP_CONTROLS:
            out.append(ch)
        elif unicodedata.category(ch) not in _DROP_CATEGORIES:
            out.append(ch)
    return "".join(out)


def clean_text(text: str) -> str:
    """Stored text (9.3): NFC, control/format characters removed except newline and tab."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return _drop_invisible(text, keep_newlines=True).strip()


# Drawing file names such as "part.svg" are not links; these suffixes are not treated as TLDs.
FILE_SUFFIXES = {"png", "jpg", "jpeg", "svg", "dxf", "dwg", "pdf", "gif", "bmp", "tif", "tiff", "webp", "txt", "json"}


def _bare_host(match: re.Match) -> str:
    token = match.group(0)
    if "/" not in token and ":" not in token and token.rsplit(".", 1)[-1].lower() in FILE_SUFFIXES:
        return token
    return LINK_REMOVED


def strip_links(text: str) -> str:
    for pattern in (_SCHEME_URL, _SCRIPT_URL, _WWW, _EMAIL):
        text = pattern.sub(LINK_REMOVED, text)
    return _BARE_HOST.sub(_bare_host, text)


def _graphemes(text: str) -> list[str]:
    """Approximate grapheme clusters: a base character plus following combining marks."""
    clusters: list[str] = []
    for ch in text:
        if clusters and (unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Mc")):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def sanitize_title(title: str | None, limit: int = PUBLIC_TITLE_MAX) -> str:
    """The public title (11): NFKC, invisible characters removed, links removed, <= 80 graphemes."""
    text = unicodedata.normalize("NFKC", title or "")
    text = _SPACES.sub(" ", text)  # whitespace controls become spaces before invisible ones are dropped
    text = _drop_invisible(text, keep_newlines=False)
    text = strip_links(text)
    text = _SPACES.sub(" ", text).strip()
    clusters = _graphemes(text)
    if len(clusters) > limit:
        text = "".join(clusters[: limit - 1]).rstrip() + "…"
    return text or UNTITLED
