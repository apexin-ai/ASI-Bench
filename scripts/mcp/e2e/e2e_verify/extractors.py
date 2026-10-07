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


_EXPORTED_FILE = re.compile(r"[^\s'\"`]+\.(?:step|stp|stl|3mf|dxf|svg)\b", re.IGNORECASE)


def _extract_exported_files(data):
    """The paths an export answer reports (build123d-mcp writes ``Exported to:`` and one
    line per file, followed by a volume/bbox echo). The answer is plain text, so the
    extractor reads the whole result."""
    if not isinstance(data, str):
        return None
    return _EXPORTED_FILE.findall(data) or None


def canon_exported_files(value):
    """File names, sorted, of a list of paths; a single path canonicalises to its own name,
    so a chained path argument can be compared against the returned list."""
    if isinstance(value, list):
        names = [canon_file_name(v) for v in value]
        return sorted(names) if names and all(names) else None
    return canon_file_name(value)


# --- listings and named-check reports -----------------------------------------

_LISTING_KEYS = ("artifacts", "files", "paths", "entries")
_PATH_FIELDS = ("path", "file", "filename", "name")


def _extract_listed_files(data):
    """Paths of a file listing: a list of path strings, or of objects carrying a
    ``path`` / ``file`` / ``filename`` / ``name`` field — either the whole result
    or the single list under an ``artifacts`` / ``files`` / ``paths`` / ``entries``
    key. Pairs with :func:`canon_exported_files`, so what is compared is the set
    of file names, not host-specific directories."""
    if isinstance(data, dict):
        lists = [data[key] for key in _LISTING_KEYS if isinstance(data.get(key), list)]
        if len(lists) != 1:
            return None
        data = lists[0]
    if not isinstance(data, list) or not data:
        return None
    out = []
    for item in data:
        if isinstance(item, str):
            out.append(item)
            continue
        if not isinstance(item, dict):
            return None
        named = [item[field] for field in _PATH_FIELDS if isinstance(item.get(field), str)]
        if not named:
            return None
        out.append(named[0])
    return out


def _extract_categorized_files(data):
    """Paths of a listing grouped by category: every string in every list under a key
    ending in ``_files`` (qe-mcp's ``qe_list_files``: ``band_files``, ``input_files``,
    ``output_files``, …). ``None`` when there is no such list or one holds a non-string."""
    if not isinstance(data, dict):
        return None
    lists = [value for key, value in sorted(data.items())
             if isinstance(key, str) and key.endswith("_files") and isinstance(value, list)]
    if not lists or any(not isinstance(item, str) or not item.strip() for value in lists for item in value):
        return None
    paths = [item for value in lists for item in value]
    return paths or None


def canon_paths(value):
    """Exact path strings: a list as its sorted, stripped strings, a single path stripped.
    For a chained argument that must be one of the paths a listing returned, verbatim."""
    if isinstance(value, list):
        out = [v.strip() for v in value if isinstance(v, str) and v.strip()]
        return sorted(out) if out and len(out) == len(value) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _extract_output_file(data):
    """The file a result says it wrote: its top-level ``filepath`` string
    (mcp-atomictoolkit's build / manipulate answers; the input file they read is a
    separate ``input_filepath``, and the ``artifacts`` list carries both, so it is
    not used). Pairs with :func:`canon_file_name`: a path argument scrubbed in the
    persisted log (``<workspace>/cu31.extxyz``) still compares by its file name."""
    if not isinstance(data, dict):
        return None
    path = data.get("filepath")
    return path if isinstance(path, str) and path.strip() else None


_GRID_SEPARATORS = re.compile(r"[,\sx×]+")


def _extract_kpoint_grid(data):
    """A Monkhorst–Pack grid: a result's ``kpoints`` field, or the value itself."""
    return data.get("kpoints") if isinstance(data, dict) else data


def canon_kpoint_grid(value):
    """``"7,7,7"`` from ``[7, 7, 7]``, ``"7,7,7"``, ``"7 7 7"``, ``"7x7x7"`` or a lone
    ``"7"`` (qe-mcp's own reading of a ``kpoints`` argument): a grid a tool suggests and
    the string the next tool is given compare equal. Anything else — ``"auto"``,
    ``"gamma"``, offsets, non-positive or fractional counts — is ``None``."""
    if isinstance(value, str):
        parts = [p for p in _GRID_SEPARATORS.split(value.strip()) if p]
        if len(parts) == 1:
            parts = parts * 3
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return None
    if len(parts) != 3:
        return None
    out = []
    for part in parts:
        if isinstance(part, bool):
            return None
        try:
            number = float(part)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number != int(number) or number < 1:
            return None
        out.append(int(number))
    return ",".join(map(str, out))


_STATUS_NAME_FIELDS = ("check", "name", "id")


def _extract_named_statuses(data):
    """``['convergence_gate:pass', ...]`` from a report whose ``checks`` list holds
    objects with a name field and a ``status``, in the report's own order."""
    items = data.get("checks") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        return None
    out = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("status"), str):
            return None
        named = [item[field] for field in _STATUS_NAME_FIELDS if isinstance(item.get(field), str)]
        if not named:
            return None
        out.append(f"{named[0]}:{item['status']}")
    return out


def canon_named_statuses(value):
    """A list of ``name:status`` strings, stripped and lower-cased. Order is kept:
    the order of a verification report is part of what a task pins."""
    if not isinstance(value, list) or not value:
        return None
    out = [v.strip().lower() for v in value if isinstance(v, str) and v.strip()]
    return out if len(out) == len(value) else None


# --- records keyed by the identifier that was requested -----------------------

def _extract_json_scalars(data):
    """Every scalar leaf of a result, at any depth.

    A REST wrapper keys its records by the identifier the caller asked for
    (``data.result.<uid>.slen`` of an E-utilities esummary), so no static dotted
    path reaches them, and the values are identifiers (an accession, a cytogenetic
    band) rather than numbers. Pairs with :func:`canon_scalar` and ``match:
    "member"``: the answer has to be one of the values the call returned."""
    out: list = []
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        elif isinstance(node, bool) or node is None:
            continue
        elif isinstance(node, (str, int, float)):
            out.append(node)
    return out or None


def canon_scalar(value):
    """One scalar as a canonical string; a list as its sorted, unique canonical strings.

    A finite number and its string spelling canonicalise together (``644`` ==
    ``"644"``, ``0.457`` == ``"0.4570"``), so a value a tool returns as a JSON number
    and an answer written as a string still compare equal; everything else is the
    stripped string (``"NP_000268.1"``, ``"12q23.2"``), compared case-sensitively.
    Anything that is not a scalar is ``None``: an answer file and a tool argument are
    agent-controlled, so a nested list or an object must read as "no value", never
    raise."""
    if isinstance(value, (list, tuple)):
        canonical = {item for item in (canon_scalar(v) for v in value) if isinstance(item, str)}
        return sorted(canonical) or None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip()
    if not text or "_" in text:                    # "1_000" is a float() literal, not a number here
        return text or None
    try:
        return str(int(text))
    except ValueError:                             # not an integer spelling: a float, or text
        pass
    try:
        number = float(text)
    except ValueError:
        return text
    return text if not math.isfinite(number) else (str(int(number)) if number == int(number)
                                                   else repr(number))


EXTRACTORS: dict[str, Extractor] = {
    "arxiv_ids": Extractor(_extract_arxiv_ids, canon_ids),
    "json_scalars": Extractor(_extract_json_scalars, canon_scalar),
    "term_counts": Extractor(_extract_term_counts, canon_counts),
    "text": Extractor(lambda data: data if isinstance(data, str) else None, canon_text),
    "rdkit_mol": Extractor(_extract_mol_field, canon_text),
    "file_name": Extractor(lambda data: data if isinstance(data, str) else None, canon_file_name),
    "exported_files": Extractor(_extract_exported_files, canon_exported_files),
    "listed_files": Extractor(_extract_listed_files, canon_exported_files),
    "named_statuses": Extractor(_extract_named_statuses, canon_named_statuses),
    "categorized_files": Extractor(_extract_categorized_files, canon_exported_files),
    "categorized_paths": Extractor(_extract_categorized_files, canon_paths),
    "kpoint_grid": Extractor(_extract_kpoint_grid, canon_kpoint_grid),
    "output_file": Extractor(_extract_output_file, canon_file_name),
}
