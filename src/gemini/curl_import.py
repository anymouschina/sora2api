"""Utilities for importing tokens from pasted curl/ecurl commands.

We only extract the Flow/Labs next-auth session cookie:
`__Secure-next-auth.session-token` (or `next-auth.session-token`).

This module intentionally ignores other cookies/headers to avoid storing
highly sensitive Google account session data.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CurlImportResult:
    cookie: Optional[str] = None
    st: Optional[str] = None


def _extract_cookie_from_curl_tokens(tokens: list[str]) -> Optional[str]:
    cookie_value: Optional[str] = None
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("-b", "--cookie"):
            if i + 1 < len(tokens):
                cookie_value = tokens[i + 1]
                i += 2
                continue
        if t in ("-H", "--header"):
            if i + 1 < len(tokens):
                header = tokens[i + 1]
                m = re.match(r"(?i)cookie\s*:\s*(.*)$", header.strip())
                if m:
                    cookie_value = m.group(1).strip()
                i += 2
                continue
        i += 1
    return cookie_value


def extract_st_from_cookie(cookie: str) -> Optional[str]:
    if not cookie:
        return None
    parts = [p.strip() for p in cookie.split(";") if p.strip() and "=" in p]
    kv = dict(p.split("=", 1) for p in parts)
    for key in ("__Secure-next-auth.session-token", "next-auth.session-token"):
        value = kv.get(key)
        if value:
            return value.strip().strip('"')
    return None


def parse_curl_or_cookie(text: str) -> CurlImportResult:
    """Parse a raw cookie string or a curl/ecurl command string.

    Returns:
        CurlImportResult(cookie=..., st=...)
    """
    raw = (text or "").strip()
    if not raw:
        return CurlImportResult()

    if raw.lower().startswith("cookie:"):
        cookie = raw.split(":", 1)[1].strip()
        return CurlImportResult(cookie=cookie, st=extract_st_from_cookie(cookie))

    # If it's already a cookie string, accept directly.
    if "curl " not in raw and "--compressed" not in raw and ("-H" not in raw and "-b" not in raw) and ";" in raw and "=" in raw:
        return CurlImportResult(cookie=raw, st=extract_st_from_cookie(raw))

    # Parse curl/ecurl command.
    try:
        tokens = shlex.split(raw, posix=True)
    except Exception:
        # fallback: treat as cookie string
        return CurlImportResult(cookie=raw, st=extract_st_from_cookie(raw))

    cookie = _extract_cookie_from_curl_tokens(tokens)
    return CurlImportResult(cookie=cookie, st=extract_st_from_cookie(cookie or ""))
