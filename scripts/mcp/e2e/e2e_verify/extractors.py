"""Named extractors for tool results that are not plain numbers.

An extractor is a pair: ``extract`` turns the parsed JSON result of one tool
call into a raw value, ``canon`` turns that raw value, or an answer / reference
/ tool-input value, into a canonical form that compares with ``==``. Both
return ``None`` for a value of the wrong shape. Specs refer to an extractor by
name (``"extract": "arxiv_ids"``); add a server-specific reader here rather
than special-casing it in the checks.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Extractor:
    extract: Callable[[Any], Any]
    canon: Callable[[Any], Any]


def _count(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number == int(number) and number >= 0 else None


# --- arXiv (ToolUniverse SMCP) ------------------------------------------------

def arxiv_id(value) -> str | None:
    """'1103.0291' from '1103.0291v1', 'arXiv:1103.0291' or an abs/pdf URL (old-style IDs kept intact)."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"^https?://(export\.)?arxiv\.org/(abs|pdf)/", "", value.strip())
    text = re.sub(r"\.pdf$", "", text).removeprefix("arXiv:")
    return re.sub(r"v\d+$", "", text) or None


def canon_ids(value):
    if isinstance(value, list):
        ids = [arxiv_id(v) for v in value]
        return ids if ids and all(ids) else None
    return arxiv_id(value)


def _extract_arxiv_ids(data):
    if isinstance(data, dict) and isinstance(data.get("data"), list):  # SMCP truncation wrapper
        data = data["data"]
    if not isinstance(data, list) or not all(isinstance(p, dict) for p in data):
        return None
    return [p.get("url") for p in data]


def canon_counts(value):
    if not isinstance(value, dict) or not value:
        return None
    out = {}
    for key, count in value.items():
        number = _count(count)
        if number is None:
            return None
        out[str(key).strip().lower()] = number
    return out


def _extract_term_counts(data):
    if not isinstance(data, dict) or data.get("status") == "error" or not isinstance(data.get("snippets"), list):
        return None
    counts: dict[str, int] = {}
    for snippet in data["snippets"]:
        if isinstance(snippet, dict) and isinstance(snippet.get("term"), str):
            counts[snippet["term"]] = counts.get(snippet["term"], 0) + 1
    return counts


# --- opaque strings and returned file paths -----------------------------------

def canon_text(value) -> str | None:
    """A non-empty string, stripped (e.g. a base64 RDKit pickle). Compared exactly; a
    copy scrubbed in the persisted log still matches (values.same_text)."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _extract_mol_field(data):
    """The ``mol`` field of an RDKit result object (EmbedMolecule / EmbedMultipleConfs)."""
    return data.get("mol") if isinstance(data, dict) else None


def canon_file_name(value) -> str | None:
    """The last component of a returned path. Host paths are scrubbed in persisted logs
    (``<workspace>/x.sdf``) and differ between runs; the file name is what a task fixes."""
    if not isinstance(value, str) or not value.strip():
        return None
    name = re.split(r"[\\/]", value.strip())[-1]
    return name or None


EXTRACTORS: dict[str, Extractor] = {
    "arxiv_ids": Extractor(_extract_arxiv_ids, canon_ids),
    "term_counts": Extractor(_extract_term_counts, canon_counts),
    "text": Extractor(lambda data: data if isinstance(data, str) else None, canon_text),
    "rdkit_mol": Extractor(_extract_mol_field, canon_text),
    "file_name": Extractor(lambda data: data if isinstance(data, str) else None, canon_file_name),
}
