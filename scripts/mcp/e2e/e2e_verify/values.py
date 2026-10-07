"""Reading values out of tool results and comparing them.

:func:`read` applies a :class:`~.spec.Selector` to one tool call; the
comparators below are shared by every check, so a new value type or
comparison is added once and is available to call results, chained inputs and
answers alike.
"""
from __future__ import annotations

import json
import math
import re

from .extractors import EXTRACTORS
from .evidence import SCRUBBED
from .spec import Selector


# --------------------------------------------------------------------------
# Persisted-log path scrubbing
# --------------------------------------------------------------------------

# The absolute-path rule of ai4sci_bench.runner.orchestrator._sanitize_persisted_text,
# which ``asibench run`` applies to every string of the saved stdout (outside
# http(s) URLs). tests/mcp_e2e keep this copy equal to the real function.
_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_>.])/(?:[^/\s'\"`]+/){1,}[^/\s'\"`]+")
_HTTP_URL = re.compile(r"https?://[^\s'\"`]+")


def scrub_host_paths(text: str) -> str:
    """``text`` as the persisted log would show it: absolute-path-like runs become
    ``<abs_path>``. Base64 data (e.g. an RDKit pickle) contains ``/`` and is hit too."""
    parts, cursor = [], 0
    for match in _HTTP_URL.finditer(text):
        parts.append(_ABSOLUTE_PATH.sub(SCRUBBED, text[cursor:match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(_ABSOLUTE_PATH.sub(SCRUBBED, text[cursor:]))
    return "".join(parts)


def same_text(given, expected) -> bool:
    """Identical non-empty strings, also when ``given`` comes from the persisted log and had
    path-like runs scrubbed: then it must equal ``expected`` scrubbed the same way (the
    unscrubbed rest still has to match character for character)."""
    if not isinstance(given, str) or not isinstance(expected, str):
        return False
    a, b = given.strip(), expected.strip()
    if not a:
        return False
    return a == b or (SCRUBBED in a and a == scrub_host_paths(b))


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def number(value) -> float | None:
    """A finite float from a number or numeric string; ``None`` otherwise (bools included)."""
    if isinstance(value, bool):
        return None
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def numbers(value):
    """A finite float, a list of finite floats, or None. Accepts numeric strings and JSON-encoded lists."""
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, list):
        out = [number(v) for v in value]
        return out if out and all(n is not None for n in out) else None
    return number(value)


def diff(a, b) -> float:
    """Max absolute difference between two scalars or two equal-length lists; inf otherwise."""
    if a is None or b is None or isinstance(a, list) != isinstance(b, list):
        return math.inf
    if isinstance(a, list):
        return max((abs(x - y) for x, y in zip(a, b)), default=0.0) if len(a) == len(b) else math.inf
    return abs(a - b)


def field(data, key):
    """Field ``key`` of a JSON object; a dotted key walks nested objects
    (``result.final_energy.value``). A literal key containing dots wins."""
    if not isinstance(data, dict) or not isinstance(key, str):
        return None
    if key in data:
        return data[key]
    node = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def geometry(text) -> list[tuple[str, tuple[float, float, float]]] | None:
    """Atom lines ``Symbol x y z`` of a geometry string; other lines (an XYZ atom count
    or comment, a psi4 ``charge multiplicity`` header, ``symmetry c1``) are ignored."""
    if not isinstance(text, str):
        return None
    atoms = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4 or not parts[0].isalpha():
            continue
        coords = [number(v) for v in parts[1:]]
        if any(c is None for c in coords):
            continue
        atoms.append((parts[0].capitalize(), tuple(coords)))
    return atoms or None


def unwrap_structured(data):
    """Undo FastMCP's structured-output wrapper.

    A FastMCP tool with a return annotation declares an ``outputSchema`` and
    returns ``structuredContent = {"result": <value>}``; clients that show the
    structured content (Claude Code does) put that object in the tool result
    instead of the text block. ``<value>`` is the original string, or for a tool
    returning content blocks a list of ``{"type": "text", "text": ...}`` blocks.
    Anything else is returned unchanged.
    """
    if not (isinstance(data, dict) and set(data) == {"result"}):
        return data
    inner = data["result"]
    if isinstance(inner, list) and inner and all(isinstance(b, dict) and "type" in b for b in inner):
        inner = "\n".join(str(b.get("text", "")) for b in inner if b.get("type") == "text")
    if isinstance(inner, str):
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            return inner
    return inner


def parsed_result(call):
    """The call's result text parsed as JSON (structured-output wrapper removed), or None."""
    try:
        return unwrap_structured(json.loads(call.result_text or ""))
    except (json.JSONDecodeError, TypeError):
        return None


def result_value(call):
    """The whole result: parsed JSON (structured-output wrapper removed), else the stripped
    text (a tool returning one plain string, e.g. a pickle, in a client that shows the text)."""
    data = parsed_result(call)
    if data is not None:
        return data
    return None if call.result_text is None else call.result_text.strip()


def json_object(call) -> dict | None:
    data = parsed_result(call)
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------
# Selectors
# --------------------------------------------------------------------------

def canon(kind: str):
    """Canonicalisation of answer / reference values of a kind: ``number`` or an extractor name."""
    return numbers if kind == "number" else EXTRACTORS[kind].canon


def readable(call, selector: Selector) -> bool:
    """Whether the call's result can serve as a source at all: a JSON object for a key
    lookup, an extracted value for an extractor, any value for the whole result."""
    if selector.extract or (selector.raw and selector.key is None):
        return read(call, selector) is not None
    return json_object(call) is not None


def read(call, selector: Selector, reference: dict | None = None):
    """The value ``selector`` reads from one tool call's result, or None.

    extract: the named extractor's canonical value. raw: field ``key`` as is (a
    chained session id or geometry). Otherwise numbers: the whole text result
    (``key`` None) or field ``key``, narrowed to one element by ``select``.
    """
    if call.result_text is None:
        return None
    if selector.extract:
        extractor = EXTRACTORS[selector.extract]
        data = result_value(call)
        raw = None if data is None else extractor.extract(data)
        return None if raw is None else extractor.canon(raw)
    if selector.raw:
        return result_value(call) if selector.key is None else field(json_object(call), selector.key)
    if selector.select is not None:
        return _select(json_object(call), selector, reference or {})
    if selector.key is None:
        value = numbers(call.result_text)
        if value is None:
            data = parsed_result(call)
            value = numbers(data) if isinstance(data, (str, int, float, list)) else None
        return value
    return numbers(field(parsed_result(call), selector.key))


def _select(data: dict | None, selector: Selector, reference: dict):
    sel = selector.select
    values = numbers(field(data, selector.key)) if data else None
    if not isinstance(values, list):
        return None
    if sel.reduce in ("max", "min"):
        return max(values) if sel.reduce == "max" else min(values)
    for other_key, pick in ((sel.argmax_of, max), (sel.argmin_of, min)):
        if other_key:
            other = numbers(field(data, other_key))
            if not isinstance(other, list) or len(other) != len(values):
                return None
            return values[other.index(pick(other))]
    if sel.where_key:
        keys = numbers(field(data, sel.where_key))
        target = number(reference.get(sel.equals_reference_key))
        if not isinstance(keys, list) or len(keys) != len(values) or target is None:
            return None
        hits = [v for k, v in zip(keys, values) if abs(k - target) <= 1e-9 * max(1.0, abs(target))]
        return hits[0] if hits else None
    return None


# --------------------------------------------------------------------------
# Comparators
# --------------------------------------------------------------------------

def within(a, b, tol: float) -> bool:
    """Numbers (or equal-length lists) within an absolute tolerance."""
    return diff(a, b) <= tol


def copied(a, b) -> bool:
    """Numbers equal up to printing precision: the value was copied, not recomputed."""
    scale = max((abs(v) for v in (a if isinstance(a, list) else [a])), default=0.0) if a is not None else 0.0
    return diff(a, b) <= max(1e-9, 1e-12 * scale)


def same_input(given, expected) -> bool:
    """A tool input equals the reference input: numbers numerically (0.9 == "0.9"),
    anything else as stripped strings."""
    a, b = numbers(given), numbers(expected)
    if a is not None and b is not None:
        return diff(a, b) <= 1e-12
    return str(given).strip() == str(expected).strip() or same_text(given, expected)


def same_link(given, source) -> bool:
    """A chained input equals an earlier result: numbers to printing precision, other
    scalars (e.g. a session id) as identical non-empty strings."""
    a, b = numbers(given), numbers(source)
    if a is not None and b is not None:
        return copied(a, b)
    if isinstance(given, (dict, list)) or isinstance(source, (dict, list)) or given is None or source is None:
        return False
    return str(given).strip() != "" and (str(given).strip() == str(source).strip() or same_text(given, source))


def same_geometry(given, source, abs_tol: float) -> bool:
    """Same atoms in the same order, every Cartesian coordinate within ``abs_tol``
    (an agent may reformat or round the geometry it passes on)."""
    a, b = geometry(given), geometry(source)
    if a is None or b is None or len(a) != len(b) or [s for s, _ in a] != [s for s, _ in b]:
        return False
    return all(abs(x - y) <= abs_tol for (_, p), (_, q) in zip(a, b) for x, y in zip(p, q))


def matches(value, ref, mode: str) -> bool:
    """Canonical values: equal, identical; subset, every key of a dict value has the
    reference count; superset, every element of a reference list is in the value list
    (a listing that also carries entries the task does not pin); member, the reference
    scalar is in the value list."""
    if value is None or ref is None:
        return False
    if mode == "member":
        return isinstance(value, list) and ref in value
    if mode == "subset":
        return isinstance(value, dict) and isinstance(ref, dict) and bool(value) and \
            all(k in ref and ref[k] == v for k, v in value.items())
    if mode == "superset":
        return isinstance(value, list) and isinstance(ref, list) and bool(ref) and \
            all(item in value for item in ref)
    return value == ref or same_text(value, ref)


def link_comparator(binding):
    """``given, source -> bool`` for one chained input (see :class:`~.spec.Binding`)."""
    if binding.compare == "geometry":
        return lambda given, source: same_geometry(given, source, binding.abs_tol)
    if binding.compare == "member":
        canonical = EXTRACTORS[binding.source.extract].canon

        def member(given, source):
            value = canonical(given)
            return value is not None and value in (source if isinstance(source, list) else [source])
        return member
    return same_link
