import re
from dataclasses import dataclass


# Matches [PDF: name, location], [Web: name, url], [YouTube: name, MM:SS]
# Location is optional — the third group may be empty if the LLM omits it.
_CITATION_RE = re.compile(
    r"\[(PDF|Web|YouTube):\s*([^,\]]+?)(?:\s*,\s*([^\]]+?))?\s*\]"
)


@dataclass(frozen=True)
class Citation:
    source_type: str  # "PDF" | "Web" | "YouTube"
    source_name: str
    location: str = ""  # "page N" | "MM:SS" | URL (empty string when LLM omits it)


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...] = ()

    @classmethod
    def from_raw(cls, raw_text: str) -> "Answer":
        """Parse [PDF/Web/YouTube: name, location] markers from raw LLM output.

        Preserves first-seen order and deduplicates identical citations.
        """
        seen: dict[tuple[str, str, str], None] = {}
        for src_type, name, loc in _CITATION_RE.findall(raw_text):
            seen.setdefault((src_type.strip(), name.strip(), loc.strip()), None)
        return cls(
            text=raw_text,
            citations=tuple(Citation(t, n, l) for t, n, l in seen),
        )
