"""Deduplicate BibTeX-like references in thesis_refs.txt by citation key.

This script reads the hardcoded input file at the repository root:
    thesis_refs.txt

It writes the deduplicated output to:
    thesis_refs_dedup.txt

For duplicate entries, the first occurrence of a key is kept and later
occurrences are removed.
"""

from __future__ import annotations

import re
from pathlib import Path


ENTRY_START_RE = re.compile(r"(?m)^[ \t]*@[A-Za-z]+\s*\{")
KEY_RE = re.compile(r"^\s*@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", re.DOTALL)


def _find_entry_end(text: str, start_index: int) -> int | None:
    """Return index right after the matching closing brace for an entry.

    Entry is assumed to start at an '@...{' pattern at ``start_index``.
    """
    open_brace_index = text.find("{", start_index)
    if open_brace_index == -1:
        return None

    depth = 0
    for i in range(open_brace_index, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def deduplicate_bibtex_like_text(text: str) -> str:
    """Remove duplicate entries by key while preserving first occurrence."""
    out_parts: list[str] = []
    seen_keys: set[str] = set()

    pos = 0
    while True:
        match = ENTRY_START_RE.search(text, pos)
        if not match:
            out_parts.append(text[pos:])
            break

        entry_start = match.start()
        out_parts.append(text[pos:entry_start])

        entry_end = _find_entry_end(text, entry_start)
        if entry_end is None:
            out_parts.append(text[entry_start:])
            break

        entry_text = text[entry_start:entry_end]
        key_match = KEY_RE.match(entry_text)

        if key_match is None:
            out_parts.append(entry_text)
        else:
            key = key_match.group(1)
            if key not in seen_keys:
                seen_keys.add(key)
                out_parts.append(entry_text)

        pos = entry_end

    return "".join(out_parts)


def main() -> None:
    # Hardcoded file names in the current working directory.
    # With your command:
    #   python scripts\data_gen_and_processing\ref_dedup.py
    # run from repo root, this reads/writes at repo root.
    input_path = Path("thesis_refs.txt")
    output_path = Path("thesis_refs_dedup.txt")

    input_text = input_path.read_text(encoding="utf-8")
    deduped_text = deduplicate_bibtex_like_text(input_text)
    output_path.write_text(deduped_text, encoding="utf-8")

    print(f"Wrote deduplicated references to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
