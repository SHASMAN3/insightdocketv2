"""
Input sanitisation and prompt injection detection.

Interview note: Prompt injection is the #1 LLM application security risk.
We apply two layers of defence:
  1. Input sanitisation — strip null bytes, HTML, truncate length
  2. Injection detection — 16+ regex patterns matching known attack vectors

Patterns are compiled once at module load for performance.
Detection returns the matched pattern name for audit logging.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import bleach
import structlog

logger = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_QUESTION_LENGTH = 2000     # Characters — truncated beyond this
MAX_DOCUMENT_NAME_LENGTH = 255


# ── Injection patterns ─────────────────────────────────────────────────────────
# Each tuple: (pattern_name, compiled_regex)
# Patterns cover: role injection, instruction override, jailbreak, leakage,
# chain-of-thought hijacking, fictional framing, and delimiter injection.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Role injection — attacker tries to override the system persona
    ("role_injection_ignore", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I)),
    ("role_injection_disregard", re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|constraints?)", re.I)),
    ("role_injection_forget", re.compile(r"forget\s+(everything|all|your\s+instructions?|your\s+role)", re.I)),

    # System prompt leakage attempts
    ("system_prompt_leak", re.compile(r"(repeat|print|output|show|reveal|display)\s+(your\s+)?(system\s+prompt|instructions?|rules?|guidelines?)", re.I)),
    ("system_prompt_leak2", re.compile(r"what\s+(are|were)\s+your\s+(original\s+)?(instructions?|rules?|constraints?|system\s+prompt)", re.I)),

    # Jailbreak — DAN, fictional framing, developer mode
    ("jailbreak_dan", re.compile(r"\bDAN\b|do\s+anything\s+now|jailbreak", re.I)),
    ("jailbreak_fictional", re.compile(r"(pretend|imagine|roleplay|act as|you are now|you're now)\s+(you\s+are\s+)?(a\s+)?(different|new|another|unrestricted|unfiltered)", re.I)),
    ("jailbreak_developer", re.compile(r"developer\s+mode|god\s+mode|unrestricted\s+mode|jailbreak\s+mode", re.I)),

    # Delimiter injection — attacker tries to inject new turns
    ("delimiter_human", re.compile(r"human\s*:", re.I)),
    ("delimiter_assistant", re.compile(r"assistant\s*:", re.I)),
    ("delimiter_system", re.compile(r"<\s*system\s*>|<<SYS>>|\[INST\]", re.I)),
    ("delimiter_xml_injection", re.compile(r"<\s*(prompt|instruction|system|context)\s*>", re.I)),

    # Instruction override via polite framing
    ("override_actually", re.compile(r"(actually|instead|rather)\s+(ignore|forget|disregard|override)", re.I)),
    ("override_new_task", re.compile(r"new\s+(task|instruction|objective|goal)\s*[:：]", re.I)),

    # Data exfiltration
    ("exfil_training_data", re.compile(r"(show|print|output|give)\s+me\s+(your\s+)?(training\s+data|dataset|examples?)", re.I)),

    # Indirect injection via PDF content (catches obvious patterns)
    ("indirect_injection", re.compile(r"when\s+(the\s+)?user\s+(asks?|says?|types?|queries?)\s+.{0,50}(respond|reply|say|answer)", re.I)),
]


@dataclass
class SanitisationResult:
    """Result of sanitising and scanning a user input string."""
    sanitised_text: str
    injection_detected: bool
    injection_pattern: Optional[str]
    original_length: int
    was_truncated: bool


def sanitise_input(text: str, max_length: int = MAX_QUESTION_LENGTH) -> SanitisationResult:
    """
    Sanitise a user-supplied string and scan for injection patterns.

    Steps:
      1. Strip null bytes and control characters
      2. Unicode normalise (NFKC) to fold homoglyph attacks
      3. HTML-escape to neutralise any markup
      4. Truncate to max_length
      5. Scan against all injection patterns

    Returns SanitisationResult with the cleaned text and detection metadata.
    """
    original_length = len(text)

    # Step 1: Strip null bytes and dangerous control characters
    # Allow \t \n \r but strip everything else below 0x20
    cleaned = "".join(
        ch for ch in text
        if ch in ("\t", "\n", "\r") or (ord(ch) >= 0x20 and ord(ch) != 0x7F)
    )

    # Step 2: Unicode normalise — folds look-alike characters
    # e.g. "ｉｇｎｏｒｅ" → "ignore"
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # Step 3: HTML escape — neutralises any injected markup
    cleaned = bleach.clean(cleaned, tags=[], strip=True)

    # Step 4: Truncate
    was_truncated = len(cleaned) > max_length
    if was_truncated:
        cleaned = cleaned[:max_length]
        logger.warning("sanitiser.truncated", original_length=original_length, max_length=max_length)

    # Step 5: Injection scan — check against all compiled patterns
    injection_pattern: Optional[str] = None
    for pattern_name, pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            injection_pattern = pattern_name
            logger.warning(
                "sanitiser.injection_detected",
                pattern=pattern_name,
                text_preview=cleaned[:100],
            )
            break  # First match is sufficient — we block the request

    return SanitisationResult(
        sanitised_text=cleaned,
        injection_detected=injection_pattern is not None,
        injection_pattern=injection_pattern,
        original_length=original_length,
        was_truncated=was_truncated,
    )


def sanitise_document_name(name: str) -> str:
    """
    Sanitise a document filename for safe storage and display.

    Strips path traversal sequences, null bytes, and non-printable chars.
    """
    # Normalise unicode
    name = unicodedata.normalize("NFKC", name)
    # Strip null bytes
    name = name.replace("\x00", "")
    # Strip path traversal
    name = name.replace("..", "").replace("/", "_").replace("\\", "_")
    # Strip non-printable (keep alphanumeric, spaces, dots, dashes, underscores)
    name = re.sub(r"[^\w\s.\-()]", "_", name)
    # Truncate
    return name[:MAX_DOCUMENT_NAME_LENGTH].strip()


def get_injection_pattern_count() -> int:
    """Return the number of compiled injection patterns (used in /metrics)."""
    return len(_INJECTION_PATTERNS)
