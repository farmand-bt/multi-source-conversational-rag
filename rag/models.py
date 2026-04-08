import re
from dataclasses import dataclass


_CITATION_RE = re.compile(r"\[Source:\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]")


@dataclass(frozen=True)
class Citation:
    source_name: str
    location: str  # "page N" | "timestamp HH:MM" | URL


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...] = ()

    @classmethod
    def from_raw(cls, raw_text: str) -> "Answer":
        """Parse [Source: name, location] markers from raw LLM output.

        Preserves first-seen order and deduplicates identical citations.
        """
        seen: dict[tuple[str, str], None] = {}
        for name, loc in _CITATION_RE.findall(raw_text):
            seen.setdefault((name.strip(), loc.strip()), None)
        return cls(
            text=raw_text,
            citations=tuple(Citation(n, l) for n, l in seen),
        )
