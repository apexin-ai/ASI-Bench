"""Exact public scorers for the refined A2-trinion and complete HL task."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


TOP_KEYS = {
    "schema_version",
    "mcp_script_name",
    "mcp_character_check_count",
    "mcp_hl_numerator_artifact",
    "mcp_hl_denominator_artifact",
    "mcp_hl_numerator_coefficients",
    "mcp_hl_denominator_coefficients",
    "full_index",
    "character_recombination",
    "hall_littlewood",
}
MCP_HL_NUMERATOR_ARTIFACT = "hl_numerator_coefficients.json"
MCP_HL_DENOMINATOR_ARTIFACT = "hl_denominator_coefficients.json"
FULL_KEYS = {"computed_through_t_power", "zero_t_powers", "terms"}
CHARACTER_KEYS = {"dynkin_label", "dimension_check", "multiplicity"}
FULL_TERM_KEYS = CHARACTER_KEYS | {"t_power", "v_power", "y_power"}
RECOMBINATION_KEYS = {
    "candidate_algebra",
    "rank",
    "dynkin_convention",
    "manifest_subgroup",
    "manifest_t4_over_v",
    "symmetric_square",
    "tensor_square",
}
MANIFEST_KEYS = {
    "t_power",
    "v_power",
    "y_power",
    "su3_cubed_terms",
    "recombined_character",
}
SU3_TERM_KEYS = {"su3_dynkin_labels", "multiplicity"}
MANIFEST_TOP_KEYS = {
    "schema_version",
    "basis",
    "computed_through_t_power",
    "zero_t_powers",
    "branchings",
    "sectors",
}
BRANCHING_KEYS = {"dynkin_label", "dimension_check", "su3_cubed_terms"}
SECTOR_KEYS = {"t_power", "v_power", "y_power", "su3_cubed_terms"}
HL_KEYS = {
    "scaling",
    "remainder",
    "terms",
    "unrefined_coefficients",
    "removed_at_tau4_characters",
    "unrefined_closed_form",
}
HL_TERM_KEYS = CHARACTER_KEYS | {"tau_power"}
UNREFINED_COEFFICIENT_KEYS = {"tau_power", "coefficient"}
HL_CLOSED_FORM_KEYS = {
    "grading_variable",
    "tau_power_per_grading_power",
    "numerator_coefficients_ascending",
    "denominator_coefficients_ascending",
}


class SubmissionError(ValueError):
    """An agent artifact is missing or malformed."""


class EvaluatorError(RuntimeError):
    """A hidden evaluator input is missing or malformed."""


def _integer(value: Any, label: str, error_type: type[Exception]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{label} must be an integer")
    return value


def _string(value: Any, label: str, error_type: type[Exception]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{label} must be a non-empty string")
    return value


def _integer_list(value: Any, label: str, error_type: type[Exception]) -> list[int]:
    if not isinstance(value, list):
        raise error_type(f"{label} must be a list")
    return [_integer(item, f"{label}[{index}]", error_type) for index, item in enumerate(value)]


def _polynomial_coefficients(
    value: Any,
    label: str,
    error_type: type[Exception],
    *,
    nonzero: bool,
) -> list[int]:
    coefficients = _integer_list(value, label, error_type)
    if not coefficients:
        raise error_type(f"{label} must be nonempty")
    if len(coefficients) > 256:
        raise error_type(f"{label} may contain at most 256 coefficients")
    if nonzero and not any(coefficients):
        raise error_type(f"{label} must not be the zero polynomial")
    return coefficients


def _trim_polynomial(coefficients: list[int]) -> list[int]:
    result = list(coefficients)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _multiply_polynomials(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, left_coefficient in enumerate(left):
        for j, right_coefficient in enumerate(right):
            result[i + j] += left_coefficient * right_coefficient
    return _trim_polynomial(result)


def _same_rational_function(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_numerator = left["numerator_coefficients_ascending"]
    left_denominator = left["denominator_coefficients_ascending"]
    right_numerator = right["numerator_coefficients_ascending"]
    right_denominator = right["denominator_coefficients_ascending"]
    return _multiply_polynomials(left_numerator, right_denominator) == _multiply_polynomials(
        right_numerator, left_denominator
    )


def _unrefined_coefficients(
    value: Any, label: str, error_type: type[Exception]
) -> dict[str, int]:
    """Validate and canonicalize the two documented exact JSON encodings.

    The canonical form is a power-to-coefficient object.  A record array is
    accepted as an equivalent serialization because JSON has no integer object
    keys and Wolfram naturally exports a list of associations.
    """
    records: list[tuple[Any, Any, str]] = []
    if isinstance(value, dict):
        records = [(order, coefficient, f"{label}.{order}") for order, coefficient in value.items()]
    elif isinstance(value, list):
        for index, record in enumerate(value):
            record_label = f"{label}[{index}]"
            if not isinstance(record, dict) or set(record) != UNREFINED_COEFFICIENT_KEYS:
                raise error_type(
                    f"{record_label} keys must be exactly {sorted(UNREFINED_COEFFICIENT_KEYS)}"
                )
            records.append((record["tau_power"], record["coefficient"], record_label))
    else:
        raise error_type(
            f"{label} must be an object or a list of tau_power/coefficient records"
        )

    canonical: dict[int, int] = {}
    for raw_order, raw_coefficient, record_label in records:
        if isinstance(value, dict):
            if not (
                isinstance(raw_order, str)
                and raw_order.isascii()
                and raw_order.isdigit()
                and str(int(raw_order)) == raw_order
            ):
                raise error_type(f"{record_label} must use a canonical non-negative integer power key")
            order = int(raw_order)
        else:
            order = _integer(raw_order, f"{record_label}.tau_power", error_type)
            if order < 0:
                raise error_type(f"{record_label}.tau_power must be non-negative")
        coefficient = _integer(raw_coefficient, f"{record_label}.coefficient", error_type)
        if order in canonical:
            raise error_type(f"{label} contains duplicate tau power {order}")
        canonical[order] = coefficient
    return {str(order): canonical[order] for order in sorted(canonical)}


def _dynkin_label(value: Any, label: str, error_type: type[Exception]) -> list[int]:
    labels = _integer_list(value, label, error_type)
    if len(labels) != 6 or any(component < 0 for component in labels):
        raise error_type(f"{label} must be six non-negative Bourbaki Dynkin labels")
    return labels


def _character(
    value: Any,
    label: str,
    error_type: type[Exception],
    *,
    exact_keys: set[str] = CHARACTER_KEYS,
    positive_multiplicity: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != exact_keys:
        raise error_type(f"{label} keys must be exactly {sorted(exact_keys)}")
    _dynkin_label(value["dynkin_label"], f"{label}.dynkin_label", error_type)
    if _integer(value["dimension_check"], f"{label}.dimension_check", error_type) <= 0:
        raise error_type(f"{label}.dimension_check must be positive")
    multiplicity = _integer(value["multiplicity"], f"{label}.multiplicity", error_type)
    if multiplicity == 0 or (positive_multiplicity and multiplicity < 0):
        qualifier = "positive" if positive_multiplicity else "nonzero"
        raise error_type(f"{label}.multiplicity must be {qualifier}")
    return value


def _load(path: Path, *, reference: bool) -> dict[str, Any]:
    error_type = EvaluatorError if reference else SubmissionError
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise error_type(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise error_type(f"{path.name} must contain a JSON object")
    return value


def _validate(payload: dict[str, Any], *, reference: bool) -> dict[str, Any]:
    error_type = EvaluatorError if reference else SubmissionError
    if set(payload) != TOP_KEYS:
        raise error_type(f"top-level keys must be exactly {sorted(TOP_KEYS)}")
    if _integer(payload["schema_version"], "schema_version", error_type) != 3:
        raise error_type("schema_version must be 3")
    if payload["mcp_script_name"] != "solution.wl":
        raise error_type("mcp_script_name must be 'solution.wl'")
    checks = _integer(payload["mcp_character_check_count"], "mcp_character_check_count", error_type)
    if not 0 <= checks <= 15:
        raise error_type("mcp_character_check_count must be between 0 and 15")
    if payload["mcp_hl_numerator_artifact"] != MCP_HL_NUMERATOR_ARTIFACT:
        raise error_type(
            f"mcp_hl_numerator_artifact must be '{MCP_HL_NUMERATOR_ARTIFACT}'"
        )
    if payload["mcp_hl_denominator_artifact"] != MCP_HL_DENOMINATOR_ARTIFACT:
        raise error_type(
            f"mcp_hl_denominator_artifact must be '{MCP_HL_DENOMINATOR_ARTIFACT}'"
        )
    mcp_hl_numerator = _polynomial_coefficients(
        payload["mcp_hl_numerator_coefficients"],
        "mcp_hl_numerator_coefficients",
        error_type,
        nonzero=False,
    )
    mcp_hl_denominator = _polynomial_coefficients(
        payload["mcp_hl_denominator_coefficients"],
        "mcp_hl_denominator_coefficients",
        error_type,
        nonzero=True,
    )
    if mcp_hl_denominator[0] == 0:
        raise error_type("mcp_hl_denominator_coefficients must have nonzero constant term")

    full = payload["full_index"]
    if not isinstance(full, dict) or set(full) != FULL_KEYS:
        raise error_type(f"full_index keys must be exactly {sorted(FULL_KEYS)}")
    _integer(full["computed_through_t_power"], "full_index.computed_through_t_power", error_type)
    _integer_list(full["zero_t_powers"], "full_index.zero_t_powers", error_type)
    terms = full["terms"]
    if not isinstance(terms, list):
        raise error_type("full_index.terms must be a list")
    identities: list[tuple[int, int, int, tuple[int, ...]]] = []
    for index, term in enumerate(terms):
        _character(term, f"full_index.terms[{index}]", error_type, exact_keys=FULL_TERM_KEYS)
        for key in ("t_power", "v_power", "y_power"):
            _integer(term[key], f"full_index.terms[{index}].{key}", error_type)
        identities.append(
            (term["t_power"], term["v_power"], term["y_power"], tuple(term["dynkin_label"]))
        )
    if len(identities) != len(set(identities)):
        raise error_type("full_index.terms contains duplicate canonical character identities")

    recombination = payload["character_recombination"]
    if not isinstance(recombination, dict) or set(recombination) != RECOMBINATION_KEYS:
        raise error_type(f"character_recombination keys must be exactly {sorted(RECOMBINATION_KEYS)}")
    for key in ("candidate_algebra", "dynkin_convention", "manifest_subgroup"):
        _string(recombination[key], f"character_recombination.{key}", error_type)
    if _integer(recombination["rank"], "character_recombination.rank", error_type) <= 0:
        raise error_type("character_recombination.rank must be positive")

    manifest = recombination["manifest_t4_over_v"]
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise error_type(f"manifest_t4_over_v keys must be exactly {sorted(MANIFEST_KEYS)}")
    for key in ("t_power", "v_power", "y_power"):
        _integer(manifest[key], f"manifest_t4_over_v.{key}", error_type)
    su3_terms = manifest["su3_cubed_terms"]
    if not isinstance(su3_terms, list) or not su3_terms:
        raise error_type("manifest_t4_over_v.su3_cubed_terms must be a non-empty list")
    for index, term in enumerate(su3_terms):
        if not isinstance(term, dict) or set(term) != SU3_TERM_KEYS:
            raise error_type(f"SU(3)^3 term {index} keys must be exactly {sorted(SU3_TERM_KEYS)}")
        labels = term["su3_dynkin_labels"]
        if not (
            isinstance(labels, list)
            and len(labels) == 3
            and all(isinstance(pair, list) and len(pair) == 2 for pair in labels)
        ):
            raise error_type(f"SU(3)^3 term {index} must contain three [p,q] label pairs")
        for factor, pair in enumerate(labels):
            for component, entry in enumerate(pair):
                if _integer(entry, f"SU(3)^3 term {index}[{factor}][{component}]", error_type) < 0:
                    raise error_type("SU(3) Dynkin labels must be non-negative")
        if _integer(term["multiplicity"], f"SU(3)^3 term {index}.multiplicity", error_type) <= 0:
            raise error_type("SU(3)^3 branching multiplicities must be positive")
    _character(
        manifest["recombined_character"],
        "manifest_t4_over_v.recombined_character",
        error_type,
        positive_multiplicity=True,
    )
    for key in ("symmetric_square", "tensor_square"):
        records = recombination[key]
        if not isinstance(records, list) or not records:
            raise error_type(f"character_recombination.{key} must be a non-empty list")
        for index, record in enumerate(records):
            _character(record, f"character_recombination.{key}[{index}]", error_type, positive_multiplicity=True)

    hl = payload["hall_littlewood"]
    if not isinstance(hl, dict) or set(hl) != HL_KEYS:
        raise error_type(f"hall_littlewood keys must be exactly {sorted(HL_KEYS)}")
    _string(hl["scaling"], "hall_littlewood.scaling", error_type)
    _string(hl["remainder"], "hall_littlewood.remainder", error_type)
    if not isinstance(hl["terms"], list):
        raise error_type("hall_littlewood.terms must be a list")
    for index, term in enumerate(hl["terms"]):
        _character(term, f"hall_littlewood.terms[{index}]", error_type, exact_keys=HL_TERM_KEYS)
        _integer(term["tau_power"], f"hall_littlewood.terms[{index}].tau_power", error_type)
    hl["unrefined_coefficients"] = _unrefined_coefficients(
        hl["unrefined_coefficients"],
        "hall_littlewood.unrefined_coefficients",
        error_type,
    )
    removed = hl["removed_at_tau4_characters"]
    if not isinstance(removed, list):
        raise error_type("removed_at_tau4_characters must be a list")
    for index, record in enumerate(removed):
        _character(record, f"removed_at_tau4_characters[{index}]", error_type, positive_multiplicity=True)
    closed = hl["unrefined_closed_form"]
    if not isinstance(closed, dict) or set(closed) != HL_CLOSED_FORM_KEYS:
        raise error_type(
            "hall_littlewood.unrefined_closed_form keys must be exactly "
            f"{sorted(HL_CLOSED_FORM_KEYS)}"
        )
    if closed["grading_variable"] != "Q":
        raise error_type("unrefined_closed_form.grading_variable must be 'Q'")
    if (
        _integer(
            closed["tau_power_per_grading_power"],
            "unrefined_closed_form.tau_power_per_grading_power",
            error_type,
        )
        != 2
    ):
        raise error_type("unrefined_closed_form.tau_power_per_grading_power must be 2")
    closed_numerator = _polynomial_coefficients(
        closed["numerator_coefficients_ascending"],
        "unrefined_closed_form.numerator_coefficients_ascending",
        error_type,
        nonzero=False,
    )
    closed_denominator = _polynomial_coefficients(
        closed["denominator_coefficients_ascending"],
        "unrefined_closed_form.denominator_coefficients_ascending",
        error_type,
        nonzero=True,
    )
    if closed_denominator[0] == 0:
        raise error_type(
            "unrefined_closed_form.denominator_coefficients_ascending must have nonzero constant term"
        )
    mcp_closed_form = {
        "numerator_coefficients_ascending": mcp_hl_numerator,
        "denominator_coefficients_ascending": mcp_hl_denominator,
    }
    if not _same_rational_function(closed, mcp_closed_form):
        raise error_type(
            "the Hall-Littlewood closed form must be exactly rationally equivalent to "
            "the canonical coefficient arrays retrieved from the Wolfram MCP job"
        )
    return payload


def _su3_terms(
    value: Any,
    label: str,
    error_type: type[Exception],
    *,
    positive_multiplicity: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise error_type(f"{label} must be a non-empty list")
    identities: list[tuple[tuple[int, int], ...]] = []
    for index, term in enumerate(value):
        term_label = f"{label}[{index}]"
        if not isinstance(term, dict) or set(term) != SU3_TERM_KEYS:
            raise error_type(f"{term_label} keys must be exactly {sorted(SU3_TERM_KEYS)}")
        labels = term["su3_dynkin_labels"]
        if not (
            isinstance(labels, list)
            and len(labels) == 3
            and all(isinstance(pair, list) and len(pair) == 2 for pair in labels)
        ):
            raise error_type(f"{term_label} must contain three [p,q] label pairs")
        identity: list[tuple[int, int]] = []
        for factor, pair in enumerate(labels):
            checked: list[int] = []
            for component, entry in enumerate(pair):
                entry = _integer(entry, f"{term_label}[{factor}][{component}]", error_type)
                if entry < 0:
                    raise error_type(f"{term_label} Dynkin labels must be non-negative")
                checked.append(entry)
            identity.append((checked[0], checked[1]))
        multiplicity = _integer(term["multiplicity"], f"{term_label}.multiplicity", error_type)
        if multiplicity == 0 or (positive_multiplicity and multiplicity < 0):
            qualifier = "positive" if positive_multiplicity else "nonzero"
            raise error_type(f"{term_label}.multiplicity must be {qualifier}")
        identities.append(tuple(identity))
    if len(identities) != len(set(identities)):
        raise error_type(f"{label} contains duplicate SU(3)^3 identities")
    return value


def _validate_manifest(payload: dict[str, Any], *, reference: bool) -> dict[str, Any]:
    error_type = EvaluatorError if reference else SubmissionError
    if set(payload) != MANIFEST_TOP_KEYS:
        raise error_type(f"manifest top-level keys must be exactly {sorted(MANIFEST_TOP_KEYS)}")
    _integer(payload["schema_version"], "manifest.schema_version", error_type)
    _string(payload["basis"], "manifest.basis", error_type)
    _integer(payload["computed_through_t_power"], "manifest.computed_through_t_power", error_type)
    _integer_list(payload["zero_t_powers"], "manifest.zero_t_powers", error_type)

    branchings = payload["branchings"]
    if not isinstance(branchings, list) or not branchings:
        raise error_type("manifest.branchings must be a non-empty list")
    branching_labels: list[tuple[int, ...]] = []
    for index, branching in enumerate(branchings):
        label = f"manifest.branchings[{index}]"
        if not isinstance(branching, dict) or set(branching) != BRANCHING_KEYS:
            raise error_type(f"{label} keys must be exactly {sorted(BRANCHING_KEYS)}")
        dynkin = _dynkin_label(branching["dynkin_label"], f"{label}.dynkin_label", error_type)
        if _integer(branching["dimension_check"], f"{label}.dimension_check", error_type) <= 0:
            raise error_type(f"{label}.dimension_check must be positive")
        _su3_terms(
            branching["su3_cubed_terms"],
            f"{label}.su3_cubed_terms",
            error_type,
            positive_multiplicity=True,
        )
        branching_labels.append(tuple(dynkin))
    if len(branching_labels) != len(set(branching_labels)):
        raise error_type("manifest.branchings contains duplicate Dynkin labels")

    sectors = payload["sectors"]
    if not isinstance(sectors, list):
        raise error_type("manifest.sectors must be a list")
    sector_identities: list[tuple[int, int, int]] = []
    for index, sector in enumerate(sectors):
        label = f"manifest.sectors[{index}]"
        if not isinstance(sector, dict) or set(sector) != SECTOR_KEYS:
            raise error_type(f"{label} keys must be exactly {sorted(SECTOR_KEYS)}")
        identity = tuple(
            _integer(sector[key], f"{label}.{key}", error_type)
            for key in ("t_power", "v_power", "y_power")
        )
        _su3_terms(
            sector["su3_cubed_terms"],
            f"{label}.su3_cubed_terms",
            error_type,
            positive_multiplicity=False,
        )
        sector_identities.append(identity)
    if len(sector_identities) != len(set(sector_identities)):
        raise error_type("manifest.sectors contains duplicate (t,v,y) sectors")
    return payload


def _paths(pred_dir: Path, ref_dir: Path, config: dict) -> tuple[Path, Path]:
    pred_name = str(config.get("result_file", "result.json"))
    ref_name = Path(str(config.get("reference_file", "result.json"))).name
    return pred_dir / pred_name, ref_dir / ref_name


def _pair(pred_dir: Path, ref_dir: Path, config: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    pred_path, ref_path = _paths(pred_dir, ref_dir, config)
    return _validate(_load(pred_path, reference=False), reference=False), _validate(
        _load(ref_path, reference=True), reference=True
    )


def _manifest_paths(pred_dir: Path, ref_dir: Path, config: dict) -> tuple[Path, Path]:
    pred_name = str(config.get("manifest_file", "manifest_index.json"))
    ref_name = Path(str(config.get("manifest_reference_file", "manifest_index.json"))).name
    return pred_dir / pred_name, ref_dir / ref_name


def _manifest_pair(
    pred_dir: Path, ref_dir: Path, config: dict
) -> tuple[dict[str, Any], dict[str, Any]]:
    pred_path, ref_path = _manifest_paths(pred_dir, ref_dir, config)
    return _validate_manifest(
        _load(pred_path, reference=False), reference=False
    ), _validate_manifest(_load(ref_path, reference=True), reference=True)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _multiset_fraction(predicted: Iterable[Any], reference: Iterable[Any]) -> float:
    pred = Counter(_canonical(value) for value in predicted)
    ref = Counter(_canonical(value) for value in reference)
    if not ref:
        return 1.0 if not pred else 0.0
    matched = sum((pred & ref).values())
    missing = sum((ref - pred).values())
    extra = sum((pred - ref).values())
    return max(0.0, (matched - 0.5 * extra) / max(matched + missing, 1))


def _coefficient_breakdown(
    predicted: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    through: int,
) -> tuple[float, dict[str, float], dict[str, bool]]:
    pred_by_order: dict[int, list[dict[str, Any]]] = {}
    ref_by_order: dict[int, list[dict[str, Any]]] = {}
    for term in predicted:
        pred_by_order.setdefault(term["t_power"], []).append(term)
    for term in reference:
        ref_by_order.setdefault(term["t_power"], []).append(term)

    orders = sorted(set(range(through + 1)) | set(pred_by_order) | set(ref_by_order))
    fractions: dict[str, float] = {}
    exact: dict[str, bool] = {}
    weighted = 0.0
    total_weight = 0
    for order in orders:
        pred_terms = pred_by_order.get(order, [])
        ref_terms = ref_by_order.get(order, [])
        fraction = _multiset_fraction(pred_terms, ref_terms)
        weight = max(len(pred_terms), len(ref_terms), 1)
        fractions[str(order)] = fraction
        exact[str(order)] = Counter(_canonical(x) for x in pred_terms) == Counter(
            _canonical(x) for x in ref_terms
        )
        weighted += fraction * weight
        total_weight += weight
    return weighted / max(total_weight, 1), fractions, exact


def _normalized_su3_terms(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(terms, key=_canonical)


def _normalized_branching(branching: dict[str, Any]) -> dict[str, Any]:
    return {
        **branching,
        "su3_cubed_terms": _normalized_su3_terms(branching["su3_cubed_terms"]),
    }


def _normalized_sector(sector: dict[str, Any]) -> dict[str, Any]:
    return {
        **sector,
        "su3_cubed_terms": _normalized_su3_terms(sector["su3_cubed_terms"]),
    }


def _su3_term_counter(terms: list[dict[str, Any]]) -> Counter:
    return Counter(
        {
            tuple(tuple(pair) for pair in term["su3_dynkin_labels"]): term["multiplicity"]
            for term in terms
        }
    )


def _counter_terms(counter: Counter) -> list[dict[str, Any]]:
    return _normalized_su3_terms(
        [
            {
                "su3_dynkin_labels": [list(pair) for pair in labels],
                "multiplicity": multiplicity,
            }
            for labels, multiplicity in counter.items()
            if multiplicity != 0
        ]
    )


def _su3_dimension(label: tuple[int, int]) -> int:
    p, q = label
    return (p + 1) * (q + 1) * (p + q + 2) // 2


def _branching_dimension_fraction(manifest: dict[str, Any]) -> tuple[float, dict[str, bool]]:
    checks: dict[str, bool] = {}
    for branching in manifest["branchings"]:
        dimension = 0
        for labels, multiplicity in _su3_term_counter(branching["su3_cubed_terms"]).items():
            dimension += multiplicity * _su3_dimension(labels[0]) * _su3_dimension(labels[1]) * _su3_dimension(labels[2])
        label = ",".join(str(component) for component in branching["dynkin_label"])
        checks[label] = dimension == branching["dimension_check"]
    return sum(checks.values()) / max(len(checks), 1), checks


def _expand_full_index(
    full_index: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    branching_by_label = {
        tuple(branching["dynkin_label"]): _su3_term_counter(branching["su3_cubed_terms"])
        for branching in manifest["branchings"]
    }
    sectors: dict[tuple[int, int, int], Counter] = {}
    for term in full_index["terms"]:
        branching = branching_by_label.get(tuple(term["dynkin_label"]), Counter())
        sector = sectors.setdefault(
            (term["t_power"], term["v_power"], term["y_power"]), Counter()
        )
        for labels, multiplicity in branching.items():
            sector[labels] += term["multiplicity"] * multiplicity
    return [
        {
            "t_power": t_power,
            "v_power": v_power,
            "y_power": y_power,
            "su3_cubed_terms": _counter_terms(terms),
        }
        for (t_power, v_power, y_power), terms in sorted(sectors.items())
        if any(multiplicity != 0 for multiplicity in terms.values())
    ]


def _exact_fields(pred: dict[str, Any], ref: dict[str, Any], fields: Iterable[str]) -> float:
    fields = tuple(fields)
    return sum(_canonical(pred.get(field)) == _canonical(ref.get(field)) for field in fields) / len(fields)


def _normalized_algebra_name(value: str) -> str:
    value = value.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    return "".join(character for character in value.casefold() if character.isalnum())


def _score(name: str, fraction: float, weight: float, details: dict[str, Any]) -> ScoreDetail:
    fraction = min(1.0, max(0.0, float(fraction)))
    return ScoreDetail(
        scorer_name=name,
        score=fraction * weight,
        max_score=weight,
        passed=fraction >= 0.5,
        details={**details, "raw_fraction": fraction},
        message=f"{name}: {fraction:.3f}",
    )


def _failure(name: str, weight: float, exc: Exception) -> ScoreDetail:
    details: dict[str, Any] = {"error": str(exc)}
    if isinstance(exc, EvaluatorError):
        details.update(scorer_internal_error=True, failure_kind="missing_evaluator_input")
    elif not isinstance(exc, SubmissionError):
        details.update(scorer_internal_error=True, failure_kind="evaluator_runtime_error")
    return ScoreDetail(
        scorer_name=name,
        score=0.0,
        max_score=weight,
        passed=False,
        details=details,
        message=f"{name} error: {exc}",
    )


@register_scorer("a2_trinion_output_contract")
class A2TrinionOutputContractScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred_path, _ = _paths(pred_dir, ref_dir, config)
            payload = _validate(_load(pred_path, reference=False), reference=False)
            manifest_path, _ = _manifest_paths(pred_dir, ref_dir, config)
            manifest = _validate_manifest(
                _load(manifest_path, reference=False), reference=False
            )
            return _score(
                "a2_trinion_output_contract",
                1.0,
                weight,
                {
                    "canonical_character_terms": len(payload["full_index"]["terms"]),
                    "manifest_branchings": len(manifest["branchings"]),
                    "manifest_sectors": len(manifest["sectors"]),
                    "mcp_character_checks": payload["mcp_character_check_count"],
                },
            )
        except Exception as exc:
            return _failure("a2_trinion_output_contract", weight, exc)


@register_scorer("a2_trinion_full_index")
class A2TrinionFullIndexScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred, ref = _pair(pred_dir, ref_dir, config)
            pfull, rfull = pred["full_index"], ref["full_index"]
            coefficients, by_order, exact_by_order = _coefficient_breakdown(
                pfull["terms"],
                rfull["terms"],
                rfull["computed_through_t_power"],
            )
            zero_orders = float(
                Counter(pfull["zero_t_powers"]) == Counter(rfull["zero_t_powers"])
            )
            truncation = float(pfull["computed_through_t_power"] == rfull["computed_through_t_power"])
            fraction = 0.96 * coefficients + 0.02 * zero_orders + 0.02 * truncation
            return _score(
                "a2_trinion_full_index",
                fraction,
                weight,
                {
                    "coefficient_character_terms": coefficients,
                    "coefficient_scores_by_t_power": by_order,
                    "coefficient_exact_by_t_power": exact_by_order,
                    "fully_matched_coefficients": sum(exact_by_order.values()),
                    "coefficient_count": len(exact_by_order),
                    "zero_orders": zero_orders,
                    "truncation": truncation,
                },
            )
        except Exception as exc:
            return _failure("a2_trinion_full_index", weight, exc)


@register_scorer("a2_trinion_manifest_index")
class A2TrinionManifestIndexScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred, ref = _manifest_pair(pred_dir, ref_dir, config)
            branchings = _multiset_fraction(
                (_normalized_branching(item) for item in pred["branchings"]),
                (_normalized_branching(item) for item in ref["branchings"]),
            )
            sectors, by_order, exact_by_order = _coefficient_breakdown(
                [_normalized_sector(item) for item in pred["sectors"]],
                [_normalized_sector(item) for item in ref["sectors"]],
                ref["computed_through_t_power"],
            )
            metadata = _exact_fields(
                pred,
                ref,
                ("computed_through_t_power", "zero_t_powers"),
            )
            dimensions, dimension_checks = _branching_dimension_fraction(pred)
            fraction = 0.32 * branchings + 0.60 * sectors + 0.03 * metadata + 0.05 * dimensions
            return _score(
                "a2_trinion_manifest_index",
                fraction,
                weight,
                {
                    "branching_characters": branchings,
                    "sector_characters": sectors,
                    "sector_scores_by_t_power": by_order,
                    "sector_exact_by_t_power": exact_by_order,
                    "fully_matched_t_powers": sum(exact_by_order.values()),
                    "t_power_count": len(exact_by_order),
                    "metadata": metadata,
                    "basis_description_present": bool(pred["basis"].strip()),
                    "branching_dimension_checks": dimension_checks,
                },
            )
        except Exception as exc:
            return _failure("a2_trinion_manifest_index", weight, exc)


@register_scorer("a2_trinion_character_recombination")
class A2TrinionCharacterRecombinationScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred, ref = _pair(pred_dir, ref_dir, config)
            pred_manifest, _ = _manifest_pair(pred_dir, ref_dir, config)
            pchar, rchar = pred["character_recombination"], ref["character_recombination"]
            algebra = float(
                _normalized_algebra_name(pchar["candidate_algebra"])
                == _normalized_algebra_name(rchar["candidate_algebra"])
            )
            rank = float(pchar["rank"] == rchar["rank"])
            invariants = 0.5 * algebra + 0.5 * rank
            pmanifest, rmanifest = pchar["manifest_t4_over_v"], rchar["manifest_t4_over_v"]
            manifest_order = _exact_fields(pmanifest, rmanifest, ("t_power", "v_power", "y_power"))
            manifest_terms = _multiset_fraction(
                pmanifest["su3_cubed_terms"], rmanifest["su3_cubed_terms"]
            )
            recombined = float(
                _canonical(pmanifest["recombined_character"])
                == _canonical(rmanifest["recombined_character"])
            )
            manifest = 0.10 * manifest_order + 0.65 * manifest_terms + 0.25 * recombined
            identities = 0.5 * (
                _multiset_fraction(pchar["symmetric_square"], rchar["symmetric_square"])
                + _multiset_fraction(pchar["tensor_square"], rchar["tensor_square"])
            )
            dimension_checks, dimension_details = _branching_dimension_fraction(pred_manifest)
            expanded_sectors = _expand_full_index(pred["full_index"], pred_manifest)
            cross_basis, cross_by_order, cross_exact = _coefficient_breakdown(
                expanded_sectors,
                [_normalized_sector(item) for item in pred_manifest["sectors"]],
                pred["full_index"]["computed_through_t_power"],
            )
            fraction = (
                0.10 * invariants
                + 0.15 * manifest
                + 0.15 * identities
                + 0.15 * dimension_checks
                + 0.45 * cross_basis
            )
            return _score(
                "a2_trinion_character_recombination",
                fraction,
                weight,
                {
                    "algebra_and_rank": invariants,
                    "algebra_symbol": algebra,
                    "rank": rank,
                    "descriptive_convention_fields_present": bool(
                        pchar["dynkin_convention"].strip()
                        and pchar["manifest_subgroup"].strip()
                    ),
                    "manifest_su3_cubed_character": manifest,
                    "representation_ring_identities": identities,
                    "branching_dimension_checks": dimension_details,
                    "full_index_manifest_cross_basis": cross_basis,
                    "cross_basis_scores_by_t_power": cross_by_order,
                    "cross_basis_exact_by_t_power": cross_exact,
                },
            )
        except Exception as exc:
            return _failure("a2_trinion_character_recombination", weight, exc)


@register_scorer("a2_trinion_hall_littlewood")
class A2TrinionHallLittlewoodScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred, ref = _pair(pred_dir, ref_dir, config)
            phl, rhl = pred["hall_littlewood"], ref["hall_littlewood"]
            # The scaling is prescribed by the instance and this field is an
            # explanatory string.  Its mathematical consequence is checked by
            # the exact character terms and coefficients below; only the
            # machine-readable truncation marker is compared literally.
            convention = float(_canonical(phl["remainder"]) == _canonical(rhl["remainder"]))
            terms = _multiset_fraction(phl["terms"], rhl["terms"])
            coefficients = float(
                _canonical(phl["unrefined_coefficients"])
                == _canonical(rhl["unrefined_coefficients"])
            )
            relations = _multiset_fraction(
                phl["removed_at_tau4_characters"], rhl["removed_at_tau4_characters"]
            )
            fraction = 0.15 * convention + 0.50 * terms + 0.20 * coefficients + 0.15 * relations
            return _score(
                "a2_trinion_hall_littlewood",
                fraction,
                weight,
                {
                    "convention": convention,
                    "scaling_description_present": bool(phl["scaling"].strip()),
                    "characters": terms,
                    "coefficients": coefficients,
                    "relations": relations,
                },
            )
        except Exception as exc:
            return _failure("a2_trinion_hall_littlewood", weight, exc)


@register_scorer("a2_trinion_hall_littlewood_closed_form")
class A2TrinionHallLittlewoodClosedFormScorer(Scorer):
    """Compare the complete unrefined HL index as an exact rational function.

    Numerator and denominator coefficient arrays are never compared separately:
    two submissions are equal precisely when exact polynomial cross
    multiplication gives the zero polynomial.  This accepts common scales,
    uncancelled common factors, and any other equivalent integer-polynomial
    presentation that is regular at the expansion origin.
    """

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred, ref = _pair(pred_dir, ref_dir, config)
            candidate = pred["hall_littlewood"]["unrefined_closed_form"]
            expected = ref["hall_littlewood"]["unrefined_closed_form"]
            exact = _same_rational_function(candidate, expected)
            return _score(
                "a2_trinion_hall_littlewood_closed_form",
                float(exact),
                weight,
                {
                    "comparison": "exact polynomial cross multiplication",
                    "exact_rational_function": exact,
                    "candidate_numerator_degree": len(
                        _trim_polynomial(candidate["numerator_coefficients_ascending"])
                    )
                    - 1,
                    "candidate_denominator_degree": len(
                        _trim_polynomial(candidate["denominator_coefficients_ascending"])
                    )
                    - 1,
                    "denominator_regular_at_origin": bool(
                        candidate["denominator_coefficients_ascending"][0]
                    ),
                },
            )
        except Exception as exc:
            return _failure("a2_trinion_hall_littlewood_closed_form", weight, exc)
