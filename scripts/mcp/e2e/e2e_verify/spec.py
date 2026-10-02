"""``e2e_check.json``: strict parsing into typed specs.

Every place that reads a value out of a tool result uses one :class:`Selector`
(``key`` dotted path, named ``extract``, list ``select``), so a selector feature
works for call results, chained inputs and answers alike. Unknown keys, invalid
enum values and dangling call references raise :class:`SpecError` when the spec
is loaded instead of silently changing what the verifier checks.

Schema 1 (one numeric tool, one scalar answer) is normalised to schema 2 first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extractors import EXTRACTORS


class SpecError(ValueError):
    """An e2e_check.json spec has unknown keys, invalid values or dangling references."""


@dataclass(frozen=True)
class SelectSpec:
    """Pick one element of a list field: ``reduce`` max/min, the element at the extreme of
    another list field (``argmax_of`` / ``argmin_of``), or the element where list field
    ``where_key`` equals the reference value ``equals_reference_key``."""
    reduce: str | None = None
    argmax_of: str | None = None
    argmin_of: str | None = None
    where_key: str | None = None
    equals_reference_key: str | None = None


@dataclass(frozen=True)
class Selector:
    """How a value is read from one tool result.

    ``extract`` applies a named extractor to the parsed JSON result; otherwise
    ``key`` (dotted path, ``None`` = the whole result) is read as numbers, or kept
    as is with ``raw`` (chained inputs such as a session id or a geometry);
    ``select`` then picks one element of that list field."""
    key: str | None = None
    extract: str | None = None
    select: SelectSpec | None = None
    raw: bool = False


@dataclass(frozen=True)
class ResultSpec:
    format: str = "number"              # number | json | image
    selector: Selector = Selector()
    reference_key: str | None = None
    abs_tol: float = 0.0
    media_type: str | None = None       # image
    match: str = "equal"                # extracted values: equal | subset | member


@dataclass(frozen=True)
class Binding:
    """One chained input: any of ``args`` of the consumer equals the value ``source``
    reads from an earlier result, compared with ``compare`` (link | geometry | member)."""
    args: tuple[str, ...]
    source: Selector
    compare: str = "link"
    abs_tol: float | None = None


@dataclass(frozen=True)
class ChainSpec:
    call: str
    bindings: tuple[Binding, ...]


@dataclass(frozen=True)
class AnswerSource:
    call: str
    selector: Selector


@dataclass(frozen=True)
class CallSpec:
    name: str
    tool: str
    inputs_from_reference: dict[str, str] = field(default_factory=dict)
    inputs_from_call: ChainSpec | None = None
    result: ResultSpec = ResultSpec()
    optional: bool = False
    group: str | None = None


@dataclass(frozen=True)
class AnswerSpec:
    prediction_key: str
    reference_key: str
    sources: tuple[AnswerSource, ...]
    kind: str = "number"                # "number" or the name of an extractor
    abs_tol: float = 0.0
    match: str = "equal"                # equal | member (prediction is one of the returned values)
    merge_calls: bool = False


@dataclass(frozen=True)
class Spec:
    schema_version: int
    server: str
    reference_file: str
    prediction_file: str
    calls: tuple[CallSpec, ...]
    answers: tuple[AnswerSpec, ...]
    server_tools: tuple[str, ...] | None = None
    bypass_patterns: tuple[str, ...] = ()
    suspicious_patterns: tuple[str, ...] = ()
    bypass_tools: dict[str, str] = field(default_factory=dict)
    suspicious_tools: dict[str, str] = field(default_factory=dict)

    def call(self, name: str) -> CallSpec:
        return next(cs for cs in self.calls if cs.name == name)

    def requirements(self) -> list[list[CallSpec]]:
        """Call specs grouped into requirements: a required spec alone, optional specs by
        ``group`` (met if any member is). An optional spec without a group is no requirement."""
        groups: dict[str, list[CallSpec]] = {}
        out: list[list[CallSpec]] = []
        for cs in self.calls:
            if not cs.optional:
                out.append([cs])
            elif cs.group:
                if cs.group not in groups:
                    groups[cs.group] = []
                    out.append(groups[cs.group])
                groups[cs.group].append(cs)
        return out


def label(requirement: list[CallSpec]) -> str:
    return "|".join(cs.tool for cs in requirement)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_TOP_KEYS = {"schema_version", "server", "reference_file", "prediction_file", "calls", "answers",
             "server_tools", "bypass_patterns", "suspicious_patterns", "bypass_tools", "suspicious_tools"}
_CALL_KEYS = {"name", "tool", "inputs_from_reference", "inputs_from_call", "result", "optional", "group"}
_RESULT_KEYS = {"format", "key", "select", "reference_key", "abs_tol", "media_type", "extract", "match"}
_CHAIN_KEYS = {"call", "map", "extract", "args", "match", "compare", "abs_tol"}
_ANSWER_KEYS = {"prediction_key", "reference_key", "abs_tol", "from_call", "result_key", "from_calls",
                "extract", "match", "merge_calls"}
_SOURCE_KEYS = {"call", "result_key", "select"}
_SELECT_KEYS = {"reduce", "argmax_of", "argmin_of", "where_key", "equals_reference_key"}

FORMATS = ("number", "json", "image")
RESULT_MATCH = ("equal", "subset", "member")
ANSWER_MATCH = ("equal", "member")
CHAIN_COMPARE = ("geometry",)


class _Parser:
    def __init__(self, path: str):
        self.path = path

    def fail(self, where: str, message: str):
        raise SpecError(f"{self.path}: {where}: {message}")

    def obj(self, data, allowed: set[str], where: str, required: tuple[str, ...] = ()) -> dict:
        if not isinstance(data, dict):
            self.fail(where, f"expected an object, got {type(data).__name__}")
        extra = set(data) - allowed
        if extra:
            self.fail(where, f"unknown key(s) {sorted(extra)}; allowed: {sorted(allowed)}")
        missing = [k for k in required if data.get(k) is None]
        if missing:
            self.fail(where, f"missing required key(s) {missing}")
        return data

    def string(self, value, where: str, *, optional: bool = True) -> str | None:
        if value is None and optional:
            return None
        if not isinstance(value, str) or not value:
            self.fail(where, f"expected a non-empty string, got {value!r}")
        return value

    def number(self, value, where: str, default: float | None = 0.0) -> float | None:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            self.fail(where, f"expected a non-negative number, got {value!r}")
        return float(value)

    def flag(self, value, where: str) -> bool:
        if value is None:
            return False
        if not isinstance(value, bool):
            self.fail(where, f"expected true/false, got {value!r}")
        return value

    def choice(self, value, choices, where: str, default: str | None = None) -> str | None:
        if value is None:
            return default
        if value not in choices:
            self.fail(where, f"{value!r} not in {sorted(choices)}")
        return value

    def str_map(self, value, where: str) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                  for k, v in value.items()):
            self.fail(where, "expected an object of string -> string")
        return dict(value)

    def str_list(self, value, where: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            self.fail(where, "expected a list of strings")
        return tuple(value)

    def regex(self, pattern: str, where: str) -> None:
        try:
            re.compile(pattern)
        except re.error as exc:
            self.fail(where, f"not a valid regex: {exc}")

    def extractor(self, value, where: str) -> str | None:
        return self.choice(value, EXTRACTORS, where)

    def select(self, data, where: str) -> SelectSpec | None:
        if data is None:
            return None
        self.obj(data, _SELECT_KEYS, where)
        sel = SelectSpec(**{k: self.string(data.get(k), f"{where}.{k}") for k in _SELECT_KEYS})
        modes = [m for m in ("reduce", "argmax_of", "argmin_of", "where_key") if getattr(sel, m)]
        if len(modes) != 1:
            self.fail(where, f"exactly one of reduce / argmax_of / argmin_of / where_key, got {modes}")
        if sel.reduce is not None:
            self.choice(sel.reduce, ("max", "min"), f"{where}.reduce")
        if (sel.where_key is None) != (sel.equals_reference_key is None):
            self.fail(where, "where_key and equals_reference_key go together")
        return sel

    # -- calls ----------------------------------------------------------------

    def result(self, data, where: str) -> ResultSpec:
        if data is None:
            return ResultSpec()
        self.obj(data, _RESULT_KEYS, where)
        fmt = self.choice(data.get("format"), FORMATS, f"{where}.format", "number")
        extract = self.extractor(data.get("extract"), f"{where}.extract")
        key = self.string(data.get("key"), f"{where}.key")
        select = self.select(data.get("select"), f"{where}.select")
        if fmt == "image":
            stray = sorted(k for k in ("key", "select", "extract", "reference_key", "abs_tol", "match")
                           if k in data)
            if stray:
                self.fail(where, f"{stray} do not apply to format 'image'")
        elif fmt == "number":
            stray = sorted(k for k in ("key", "select", "extract", "media_type") if k in data)
            if stray:
                self.fail(where, f"{stray} do not apply to format 'number' (use format 'json')")
        else:
            if "media_type" in data:
                self.fail(where, "media_type only applies to format 'image'")
            if key is None and extract is None:
                self.fail(where, "format 'json' requires 'key' or 'extract'")
            if extract is not None and (key is not None or select is not None):
                self.fail(where, "'extract' reads the whole result; drop 'key' / 'select'")
        if fmt != "image" and data.get("reference_key") is None:
            self.fail(where, "missing required key 'reference_key'")
        if "match" in data and extract is None:
            self.fail(where, "'match' only applies to extracted results")
        return ResultSpec(
            format=fmt, selector=Selector(key=key, extract=extract, select=select),
            reference_key=self.string(data.get("reference_key"), f"{where}.reference_key"),
            abs_tol=self.number(data.get("abs_tol"), f"{where}.abs_tol"),
            media_type=self.string(data.get("media_type"), f"{where}.media_type"),
            match=self.choice(data.get("match"), RESULT_MATCH, f"{where}.match", "equal"))

    def chain(self, data, where: str) -> ChainSpec | None:
        if data is None:
            return None
        self.obj(data, _CHAIN_KEYS, where, required=("call",))
        call = self.string(data["call"], f"{where}.call", optional=False)
        extract = self.extractor(data.get("extract"), f"{where}.extract")
        if extract is not None:
            stray = sorted(k for k in ("map", "compare", "abs_tol") if k in data)
            if stray:
                self.fail(where, f"{stray} do not apply to an 'extract' link (use 'args')")
            args = self.str_list(data.get("args"), f"{where}.args")
            if not args:
                self.fail(where, "an 'extract' link needs 'args'")
            self.choice(data.get("match"), ("member",), f"{where}.match")
            return ChainSpec(call, (Binding(args, Selector(extract=extract), "member"),))
        stray = sorted(k for k in ("args", "match") if k in data)
        if stray:
            self.fail(where, f"{stray} only apply to an 'extract' link")
        mapping = self.str_map(data.get("map"), f"{where}.map")
        if not mapping:
            self.fail(where, "needs 'map' (consumer argument -> source result key) or 'extract'")
        compare = self.choice(data.get("compare"), CHAIN_COMPARE, f"{where}.compare", "link")
        if "abs_tol" in data and compare != "geometry":
            self.fail(where, "abs_tol only applies to compare 'geometry'")
        tol = self.number(data.get("abs_tol"), f"{where}.abs_tol", 1e-4) if compare == "geometry" else None
        return ChainSpec(call, tuple(Binding((arg,), Selector(key=key, raw=True), compare, tol)
                                     for arg, key in mapping.items()))

    def call_spec(self, data, where: str) -> CallSpec:
        self.obj(data, _CALL_KEYS, where, required=("name", "tool"))
        return CallSpec(
            name=self.string(data["name"], f"{where}.name", optional=False),
            tool=self.string(data["tool"], f"{where}.tool", optional=False),
            inputs_from_reference=self.str_map(data.get("inputs_from_reference"), f"{where}.inputs_from_reference"),
            inputs_from_call=self.chain(data.get("inputs_from_call"), f"{where}.inputs_from_call"),
            result=self.result(data.get("result"), f"{where}.result"),
            optional=self.flag(data.get("optional"), f"{where}.optional"),
            group=self.string(data.get("group"), f"{where}.group"))

    # -- answers --------------------------------------------------------------

    def answer(self, data, where: str) -> AnswerSpec:
        self.obj(data, _ANSWER_KEYS, where, required=("prediction_key", "reference_key"))
        kind = self.extractor(data.get("extract"), f"{where}.extract") or "number"
        match = self.choice(data.get("match"), ANSWER_MATCH, f"{where}.match", "equal")
        if kind == "number" and "match" in data:
            self.fail(where, "'match' only applies to extracted answers")
        if kind != "number" and "abs_tol" in data:
            self.fail(where, "abs_tol only applies to numeric answers")
        merge = self.flag(data.get("merge_calls"), f"{where}.merge_calls")
        if merge and kind == "number":
            self.fail(where, "merge_calls only applies to extracted answers")
        if (data.get("from_call") is None) == (data.get("from_calls") is None):
            self.fail(where, "exactly one of from_call / from_calls")
        if data.get("from_call") is not None:
            sources = ((self.string(data["from_call"], f"{where}.from_call", optional=False),
                        self.string(data.get("result_key"), f"{where}.result_key"), None),)
        else:
            if "result_key" in data:
                self.fail(where, "result_key goes inside each from_calls entry")
            if not isinstance(data["from_calls"], list) or not data["from_calls"]:
                self.fail(f"{where}.from_calls", "expected a non-empty list")
            sources = []
            for k, src in enumerate(data["from_calls"]):
                sw = f"{where}.from_calls[{k}]"
                self.obj(src, _SOURCE_KEYS, sw, required=("call",))
                sel = self.select(src.get("select"), f"{sw}.select")
                key = self.string(src.get("result_key"), f"{sw}.result_key")
                if sel is not None and key is None:
                    self.fail(sw, "'select' needs the list field 'result_key'")
                sources.append((self.string(src["call"], f"{sw}.call", optional=False), key, sel))
        if kind != "number" and any(key is not None or sel is not None for _, key, sel in sources):
            self.fail(where, "an extracted answer reads the whole result; drop 'result_key' / 'select'")
        extract = None if kind == "number" else kind
        return AnswerSpec(
            prediction_key=self.string(data["prediction_key"], f"{where}.prediction_key", optional=False),
            reference_key=self.string(data["reference_key"], f"{where}.reference_key", optional=False),
            sources=tuple(AnswerSource(call, Selector(key=key, extract=extract, select=sel))
                          for call, key, sel in sources),
            kind=kind, abs_tol=self.number(data.get("abs_tol"), f"{where}.abs_tol"),
            match=match, merge_calls=merge)

    # -- whole spec -----------------------------------------------------------

    def spec(self, raw) -> Spec:
        if not isinstance(raw, dict):
            self.fail("top-level", "expected an object")
        data = normalize_spec(raw)
        self.obj(data, _TOP_KEYS, "top-level",
                 required=("server", "reference_file", "prediction_file", "calls", "answers"))
        for key in ("calls", "answers"):
            if not isinstance(data[key], list) or not data[key]:
                self.fail(key, "expected a non-empty list")
        calls = tuple(self.call_spec(cs, f"calls[{i}] ({cs.get('name', '?') if isinstance(cs, dict) else '?'})")
                      for i, cs in enumerate(data["calls"]))
        answers = tuple(self.answer(a, f"answers[{j}]") for j, a in enumerate(data["answers"]))

        names = [cs.name for cs in calls]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            self.fail("calls", f"duplicate call name(s) {dupes}")
        by_name = {cs.name: cs for cs in calls}
        for i, cs in enumerate(calls):
            if cs.group and not cs.optional:
                self.fail(f"calls[{i}] ({cs.name})", "'group' only applies to optional calls")
            link = cs.inputs_from_call
            if link is None:
                continue
            where = f"calls[{i}] ({cs.name}).inputs_from_call.call"
            if link.call not in by_name:
                self.fail(where, f"{link.call!r} references an undefined call; defined: {sorted(by_name)}")
            if link.call == cs.name:
                self.fail(where, "a call cannot chain from itself")
            if not cs.optional and by_name[link.call].optional:
                self.fail(where, f"required call {cs.name!r} chains from optional call {link.call!r}")
        for j, ans in enumerate(answers):
            for k, src in enumerate(ans.sources):
                if src.call not in by_name:
                    self.fail(f"answers[{j}] source {k} ({ans.prediction_key})",
                              f"from_call {src.call!r} references an undefined call; defined: {sorted(by_name)}")

        bypass = self.str_list(data.get("bypass_patterns"), "bypass_patterns")
        suspicious = self.str_list(data.get("suspicious_patterns"), "suspicious_patterns")
        for name, patterns in (("bypass_patterns", bypass), ("suspicious_patterns", suspicious)):
            for idx, pattern in enumerate(patterns):
                self.regex(pattern, f"{name}[{idx}]")
        tools = {}
        for name in ("bypass_tools", "suspicious_tools"):
            tools[name] = self.str_map(data.get(name), name)
            for tool, pattern in tools[name].items():
                self.regex(pattern, f"{name}[{tool!r}]")
        server_tools = data.get("server_tools")
        return Spec(
            schema_version=int(data.get("schema_version", 2)),
            server=self.string(data["server"], "server", optional=False),
            reference_file=self.string(data["reference_file"], "reference_file", optional=False),
            prediction_file=self.string(data["prediction_file"], "prediction_file", optional=False),
            calls=calls, answers=answers,
            server_tools=None if server_tools is None else self.str_list(server_tools, "server_tools"),
            bypass_patterns=bypass, suspicious_patterns=suspicious,
            bypass_tools=tools["bypass_tools"], suspicious_tools=tools["suspicious_tools"])


def normalize_spec(spec: dict) -> dict:
    """Return a schema-2 spec dict; schema 1 is one numeric-result tool and one scalar answer."""
    if int(spec.get("schema_version", 1)) >= 2:
        return spec
    tol = float(spec["abs_tol"])
    return {
        "schema_version": 2,
        "server": spec["server"],
        "reference_file": spec["reference_file"],
        "prediction_file": spec["prediction_file"],
        "calls": [{"name": spec["tool"], "tool": spec["tool"],
                   "inputs_from_reference": spec.get("tool_inputs_from_reference") or {},
                   "result": {"format": "number", "reference_key": spec["reference_key"], "abs_tol": tol}}],
        "answers": [{"prediction_key": spec["prediction_key"], "from_call": spec["tool"],
                     "reference_key": spec["reference_key"], "abs_tol": tol}],
        "bypass_patterns": spec.get("bypass_patterns", []),
        "suspicious_patterns": spec.get("suspicious_patterns", []),
    }


def parse_spec(raw: dict, path: str = "<spec>") -> Spec:
    """Validate a raw ``e2e_check.json`` document and return a :class:`Spec`."""
    return _Parser(path).spec(raw)
