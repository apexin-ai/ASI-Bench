"""Exact scorers for the affine sl(4) singular-vector task.

The scorer treats three presentational freedoms mathematically:

* affine-mode words are reduced to a private PBW order before comparison;
* the singular vector is compared only up to a non-zero scalar;
* Harish--Chandra polynomials are compared by their rational row space, and
  highest-weight families are compared as affine lines, independently of the
  parameter name or affine reparametrization.

No submitted Wolfram Language code is executed by the evaluator.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


class SubmissionError(ValueError):
    """An agent artifact is absent or malformed."""


class EvaluatorError(RuntimeError):
    """A hidden evaluator artifact is absent or malformed."""


RATIONAL_RE = re.compile(r"^[+-]?\d+(?:/[1-9]\d*)?$")
TOP_KEYS = {
    "schema_version",
    "mcp_script_name",
    "mcp_check_count",
    "mcp_search_artifact",
    "mcp_singular_artifact",
    "mcp_classification_artifact",
    "mcp_search_certificate",
    "mcp_singular_certificate",
    "mcp_classification_certificate",
    "search",
    "singular_vector",
    "zhu",
    "category_o",
}
MCP_CHECK_COUNT = 44
MCP_SEARCH_ARTIFACT = "search_certificate.json"
MCP_SINGULAR_ARTIFACT = "singular_certificate.json"
MCP_CLASSIFICATION_ARTIFACT = "classification_certificate.json"
SEARCH_KEYS = {"tested_weights", "first_nonvacuum_conformal_weight"}
SEARCH_ROW_KEYS = {"conformal_weight", "singular_nullity"}
SEARCH_CERTIFICATE_KEYS = {
    "tested_weights",
    "first_nonvacuum_conformal_weight",
    "finite_highest_weight_dynkin",
    "solution_space_dimension",
}
VECTOR_KEYS = {
    "finite_highest_weight_dynkin",
    "solution_space_dimension",
    "normalization",
    "terms",
    "annihilation_residual_term_counts",
}
SINGULAR_CERTIFICATE_KEYS = {
    "conformal_weight",
    "finite_highest_weight_dynkin",
    "solution_space_dimension",
    "nonzero",
    "homogeneous",
    "annihilation_true_count",
    "annihilation_residual_term_counts",
}
TERM_KEYS = {"coefficient", "factors"}
ZHU_KEYS = {"zero_weight_space_dimension", "polynomial_basis"}
POLYNOMIAL_KEYS = {"name", "terms"}
POLY_TERM_KEYS = {"coefficient", "exponents"}
CATEGORY_KEYS = {"parameter", "families"}
FAMILY_KEYS = {"label", "offset", "direction"}
CLASSIFICATION_CERTIFICATE_KEYS = {
    "zero_weight_space_dimension",
    "polynomial_space_rank",
    "family_count",
    "distinct_family_count",
    "substitution_true_count",
    "substitution_check_count",
    "decomposition_complete",
}


def _integer(value: Any, label: str, error_type: type[Exception]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{label} must be an integer")
    return value


def _rational(value: Any, label: str, error_type: type[Exception]) -> Fraction:
    if not isinstance(value, str) or len(value) > 80 or not RATIONAL_RE.fullmatch(value):
        raise error_type(f"{label} must be an exact integer or rational string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise error_type(f"{label} is not a valid rational number") from exc


def _load(path: Path, *, reference: bool) -> dict[str, Any]:
    error_type = EvaluatorError if reference else SubmissionError
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise error_type(f"missing {path.name}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise error_type(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise error_type(f"{path.name} must contain a JSON object")
    return payload


def _factor(value: Any, label: str, error_type: type[Exception]) -> tuple[str, int, int, int]:
    if not isinstance(value, dict):
        raise error_type(f"{label} must be an object")
    generator = value.get("generator")
    if generator == "h":
        if set(value) != {"generator", "i", "mode"}:
            raise error_type(f"{label} has invalid Cartan-factor keys")
        i = _integer(value["i"], f"{label}.i", error_type)
        if i not in (1, 2, 3):
            raise error_type(f"{label}.i must be 1, 2, or 3")
        j = 0
    elif generator in ("e", "f"):
        if set(value) != {"generator", "i", "j", "mode"}:
            raise error_type(f"{label} has invalid root-factor keys")
        i = _integer(value["i"], f"{label}.i", error_type)
        j = _integer(value["j"], f"{label}.j", error_type)
        if not (1 <= i < j <= 4):
            raise error_type(f"{label} requires 1 <= i < j <= 4")
    else:
        raise error_type(f"{label}.generator must be e, f, or h")
    mode = _integer(value.get("mode"), f"{label}.mode", error_type)
    if mode >= 0:
        raise error_type(f"{label}.mode must be negative")
    return str(generator), i, j, mode


def _terms(
    value: Any,
    label: str,
    error_type: type[Exception],
) -> list[tuple[Fraction, tuple[tuple[str, int, int, int], ...]]]:
    if not isinstance(value, list) or not value or len(value) > 300:
        raise error_type(f"{label} must be a non-empty list of at most 300 terms")
    out: list[tuple[Fraction, tuple[tuple[str, int, int, int], ...]]] = []
    for index, term in enumerate(value):
        term_label = f"{label}[{index}]"
        if not isinstance(term, dict) or set(term) != TERM_KEYS:
            raise error_type(f"{term_label} keys must be exactly {sorted(TERM_KEYS)}")
        coefficient = _rational(term["coefficient"], f"{term_label}.coefficient", error_type)
        factors = term["factors"]
        if not isinstance(factors, list) or not factors or len(factors) > 8:
            raise error_type(f"{term_label}.factors must contain 1--8 factors")
        word = tuple(
            _factor(factor, f"{term_label}.factors[{factor_index}]", error_type)
            for factor_index, factor in enumerate(factors)
        )
        out.append((coefficient, word))
    return out


def _polynomial(
    value: Any,
    label: str,
    error_type: type[Exception],
) -> dict[tuple[int, int, int], Fraction]:
    if not isinstance(value, dict) or set(value) != POLYNOMIAL_KEYS:
        raise error_type(f"{label} keys must be exactly {sorted(POLYNOMIAL_KEYS)}")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise error_type(f"{label}.name must be a non-empty string")
    terms = value["terms"]
    if not isinstance(terms, list) or not terms or len(terms) > 200:
        raise error_type(f"{label}.terms must be a non-empty list of at most 200 terms")
    out: defaultdict[tuple[int, int, int], Fraction] = defaultdict(Fraction)
    for index, term in enumerate(terms):
        term_label = f"{label}.terms[{index}]"
        if not isinstance(term, dict) or set(term) != POLY_TERM_KEYS:
            raise error_type(f"{term_label} keys must be exactly {sorted(POLY_TERM_KEYS)}")
        exponents = term["exponents"]
        if not (
            isinstance(exponents, list)
            and len(exponents) == 3
            and all(isinstance(item, int) and not isinstance(item, bool) for item in exponents)
            and all(0 <= item <= 20 for item in exponents)
        ):
            raise error_type(f"{term_label}.exponents must be three non-negative integers")
        coefficient = _rational(term["coefficient"], f"{term_label}.coefficient", error_type)
        out[tuple(exponents)] += coefficient
    return {monomial: coefficient for monomial, coefficient in out.items() if coefficient}


def _family(
    value: Any,
    label: str,
    error_type: type[Exception],
) -> tuple[tuple[Fraction, Fraction, Fraction], tuple[Fraction, Fraction, Fraction]]:
    if not isinstance(value, dict) or set(value) != FAMILY_KEYS:
        raise error_type(f"{label} keys must be exactly {sorted(FAMILY_KEYS)}")
    if not isinstance(value["label"], str) or not value["label"].strip():
        raise error_type(f"{label}.label must be a non-empty string")
    parsed: list[tuple[Fraction, Fraction, Fraction]] = []
    for key in ("offset", "direction"):
        entries = value[key]
        if not isinstance(entries, list) or len(entries) != 3:
            raise error_type(f"{label}.{key} must contain three rational strings")
        parsed.append(
            tuple(
                _rational(entry, f"{label}.{key}[{index}]", error_type)
                for index, entry in enumerate(entries)
            )
        )
    if not any(parsed[1]):
        raise error_type(f"{label}.direction must be non-zero")
    return parsed[0], parsed[1]


def _validate(payload: dict[str, Any], *, reference: bool) -> dict[str, Any]:
    error_type = EvaluatorError if reference else SubmissionError
    if set(payload) != TOP_KEYS:
        raise error_type(f"top-level keys must be exactly {sorted(TOP_KEYS)}")
    if _integer(payload["schema_version"], "schema_version", error_type) != 2:
        raise error_type("schema_version must equal 2")
    if payload["mcp_script_name"] != "solution.wl":
        raise error_type("mcp_script_name must be 'solution.wl'")
    checks = _integer(payload["mcp_check_count"], "mcp_check_count", error_type)
    if not 0 <= checks <= MCP_CHECK_COUNT:
        raise error_type(f"mcp_check_count must be between 0 and {MCP_CHECK_COUNT}")
    artifact_contract = {
        "mcp_search_artifact": MCP_SEARCH_ARTIFACT,
        "mcp_singular_artifact": MCP_SINGULAR_ARTIFACT,
        "mcp_classification_artifact": MCP_CLASSIFICATION_ARTIFACT,
    }
    for field, expected in artifact_contract.items():
        if payload[field] != expected:
            raise error_type(f"{field} must be '{expected}'")

    search = payload["search"]
    if not isinstance(search, dict) or set(search) != SEARCH_KEYS:
        raise error_type(f"search keys must be exactly {sorted(SEARCH_KEYS)}")
    first_weight = _integer(
        search["first_nonvacuum_conformal_weight"],
        "search.first_nonvacuum_conformal_weight",
        error_type,
    )
    if first_weight <= 0:
        raise error_type("the first non-vacuum conformal weight must be positive")
    rows = search["tested_weights"]
    if not isinstance(rows, list) or not rows or len(rows) > 30:
        raise error_type("search.tested_weights must be a non-empty list")
    seen_weights: set[int] = set()
    for index, row in enumerate(rows):
        label = f"search.tested_weights[{index}]"
        if not isinstance(row, dict) or set(row) != SEARCH_ROW_KEYS:
            raise error_type(f"{label} keys must be exactly {sorted(SEARCH_ROW_KEYS)}")
        weight = _integer(row["conformal_weight"], f"{label}.conformal_weight", error_type)
        nullity = _integer(row["singular_nullity"], f"{label}.singular_nullity", error_type)
        if weight <= 0 or nullity < 0 or weight in seen_weights:
            raise error_type(f"{label} contains an invalid or duplicate weight")
        seen_weights.add(weight)

    vector = payload["singular_vector"]
    if not isinstance(vector, dict) or set(vector) != VECTOR_KEYS:
        raise error_type(f"singular_vector keys must be exactly {sorted(VECTOR_KEYS)}")
    dynkin = vector["finite_highest_weight_dynkin"]
    if not (
        isinstance(dynkin, list)
        and len(dynkin) == 3
        and all(isinstance(item, int) and not isinstance(item, bool) for item in dynkin)
    ):
        raise error_type("finite_highest_weight_dynkin must contain three integers")
    if _integer(
        vector["solution_space_dimension"],
        "singular_vector.solution_space_dimension",
        error_type,
    ) <= 0:
        raise error_type("solution_space_dimension must be positive")
    if not isinstance(vector["normalization"], str) or not vector["normalization"].strip():
        raise error_type("singular_vector.normalization must be a non-empty string")
    _terms(vector["terms"], "singular_vector.terms", error_type)
    residuals = vector["annihilation_residual_term_counts"]
    if not isinstance(residuals, dict) or set(residuals) != {"e12_0", "e23_0", "e34_0", "f14_1"}:
        raise error_type("annihilation_residual_term_counts has invalid keys")
    for name, count in residuals.items():
        if _integer(count, f"annihilation_residual_term_counts.{name}", error_type) < 0:
            raise error_type("annihilation residual counts must be non-negative")

    zhu = payload["zhu"]
    if not isinstance(zhu, dict) or set(zhu) != ZHU_KEYS:
        raise error_type(f"zhu keys must be exactly {sorted(ZHU_KEYS)}")
    if _integer(zhu["zero_weight_space_dimension"], "zhu.zero_weight_space_dimension", error_type) <= 0:
        raise error_type("zero_weight_space_dimension must be positive")
    polynomials = zhu["polynomial_basis"]
    if not isinstance(polynomials, list) or not polynomials or len(polynomials) > 10:
        raise error_type("zhu.polynomial_basis must be a non-empty list")
    parsed_polynomials = [
        _polynomial(polynomial, f"zhu.polynomial_basis[{index}]", error_type)
        for index, polynomial in enumerate(polynomials)
    ]

    category = payload["category_o"]
    if not isinstance(category, dict) or set(category) != CATEGORY_KEYS:
        raise error_type(f"category_o keys must be exactly {sorted(CATEGORY_KEYS)}")
    if not isinstance(category["parameter"], str) or not category["parameter"].strip():
        raise error_type("category_o.parameter must be a non-empty string")
    families = category["families"]
    if not isinstance(families, list) or not families or len(families) > 100:
        raise error_type("category_o.families must be a non-empty list")
    labels: set[str] = set()
    parsed_families = []
    for index, family in enumerate(families):
        parsed_families.append(_family(family, f"category_o.families[{index}]", error_type))
        family_label = family["label"]
        if family_label in labels:
            raise error_type("category_o family labels must be unique")
        labels.add(family_label)

    search_certificate = payload["mcp_search_certificate"]
    if not isinstance(search_certificate, dict) or set(search_certificate) != SEARCH_CERTIFICATE_KEYS:
        raise error_type(
            f"mcp_search_certificate keys must be exactly {sorted(SEARCH_CERTIFICATE_KEYS)}"
        )
    expected_search_certificate = {
        "tested_weights": rows,
        "first_nonvacuum_conformal_weight": first_weight,
        "finite_highest_weight_dynkin": dynkin,
        "solution_space_dimension": vector["solution_space_dimension"],
    }
    if search_certificate != expected_search_certificate:
        raise error_type("mcp_search_certificate must match the submitted search result")

    canonical_vector = _canonical_vector(_terms(vector["terms"], "singular_vector.terms", error_type))
    degrees, weights = _degree_and_weights(canonical_vector)
    action_details = {
        name: len(_act_vector(canonical_vector, generator, mode))
        for name, generator, mode in SINGULAR_GENERATORS
    }
    if residuals != action_details:
        raise error_type(
            "singular_vector.annihilation_residual_term_counts must match exact recomputation"
        )
    singular_certificate = payload["mcp_singular_certificate"]
    if not isinstance(singular_certificate, dict) or set(singular_certificate) != SINGULAR_CERTIFICATE_KEYS:
        raise error_type(
            f"mcp_singular_certificate keys must be exactly {sorted(SINGULAR_CERTIFICATE_KEYS)}"
        )
    expected_singular_certificate = {
        "conformal_weight": first_weight,
        "finite_highest_weight_dynkin": dynkin,
        "solution_space_dimension": vector["solution_space_dimension"],
        "nonzero": bool(canonical_vector),
        "homogeneous": degrees == {first_weight} and weights == {tuple(dynkin)},
        "annihilation_true_count": sum(count == 0 for count in action_details.values()),
        "annihilation_residual_term_counts": action_details,
    }
    if singular_certificate != expected_singular_certificate:
        raise error_type("mcp_singular_certificate must match the submitted singular-vector computation")

    monomials = sorted(set().union(*(set(polynomial) for polynomial in parsed_polynomials)))
    polynomial_rank = len(_polynomial_space(parsed_polynomials, monomials))
    distinct_family_count = len({_line_key(family) for family in parsed_families})
    substitution_checks = [
        _line_satisfies_polynomial(family, polynomial)
        for family in parsed_families
        for polynomial in parsed_polynomials
    ]
    classification_certificate = payload["mcp_classification_certificate"]
    if (
        not isinstance(classification_certificate, dict)
        or set(classification_certificate) != CLASSIFICATION_CERTIFICATE_KEYS
    ):
        raise error_type(
            "mcp_classification_certificate keys must be exactly "
            f"{sorted(CLASSIFICATION_CERTIFICATE_KEYS)}"
        )
    expected_classification_certificate = {
        "zero_weight_space_dimension": zhu["zero_weight_space_dimension"],
        "polynomial_space_rank": polynomial_rank,
        "family_count": len(parsed_families),
        "distinct_family_count": distinct_family_count,
        "substitution_true_count": sum(substitution_checks),
        "substitution_check_count": len(substitution_checks),
        "decomposition_complete": True,
    }
    if classification_certificate != expected_classification_certificate:
        raise error_type(
            "mcp_classification_certificate must match the submitted Zhu/category-O computation"
        )
    return payload


# ---------------------------------------------------------------------------
# Exact affine PBW algebra
# ---------------------------------------------------------------------------

Base = tuple[str, int, int]
Token = tuple[str, int, int, int]
Word = tuple[Token, ...]

BASE_ORDER: tuple[Base, ...] = (
    ("e", 1, 2), ("e", 1, 3), ("e", 1, 4),
    ("e", 2, 3), ("e", 2, 4), ("e", 3, 4),
    ("h", 1, 0), ("h", 2, 0), ("h", 3, 0),
    ("f", 1, 2), ("f", 1, 3), ("f", 1, 4),
    ("f", 2, 3), ("f", 2, 4), ("f", 3, 4),
)
BASE_INDEX = {base: index for index, base in enumerate(BASE_ORDER)}
LEVEL = Fraction(-5, 2)


def _base(token: Token) -> Base:
    return token[0], token[1], token[2]


def _token(base: Base, mode: int) -> Token:
    return base[0], base[1], base[2], mode


def _token_key(token: Token) -> tuple[int, int]:
    return token[3], BASE_INDEX[_base(token)]


@lru_cache(maxsize=None)
def _matrix(base: Base) -> tuple[int, ...]:
    kind, i, j = base
    values = [0] * 16
    if kind == "e":
        values[(i - 1) * 4 + (j - 1)] = 1
    elif kind == "f":
        values[(j - 1) * 4 + (i - 1)] = 1
    else:
        values[(i - 1) * 4 + (i - 1)] = 1
        values[i * 4 + i] = -1
    return tuple(values)


def _matmul(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sum(left[row * 4 + inner] * right[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    )


@lru_cache(maxsize=None)
def _bracket(left: Base, right: Base) -> tuple[tuple[Base, Fraction], ...]:
    product_lr = _matmul(_matrix(left), _matrix(right))
    product_rl = _matmul(_matrix(right), _matrix(left))
    matrix = tuple(a - b for a, b in zip(product_lr, product_rl))
    out: dict[Base, Fraction] = {}
    for i in range(4):
        for j in range(4):
            if i == j or matrix[i * 4 + j] == 0:
                continue
            base = ("e", i + 1, j + 1) if i < j else ("f", j + 1, i + 1)
            out[base] = Fraction(matrix[i * 4 + j])
    diagonal = [matrix[i * 4 + i] for i in range(4)]
    if sum(diagonal) != 0:
        raise RuntimeError("sl(4) bracket acquired nonzero trace")
    cartan_coefficients = [diagonal[0], diagonal[0] + diagonal[1], sum(diagonal[:3])]
    for i, coefficient in enumerate(cartan_coefficients, start=1):
        if coefficient:
            out[("h", i, 0)] = Fraction(coefficient)
    return tuple(sorted(out.items(), key=lambda item: BASE_INDEX[item[0]]))


@lru_cache(maxsize=None)
def _bilinear(left: Base, right: Base) -> Fraction:
    product = _matmul(_matrix(left), _matrix(right))
    return Fraction(sum(product[i * 4 + i] for i in range(4)))


def _combine(target: defaultdict[Word, Fraction], source: Iterable[tuple[Word, Fraction]], scale: Fraction = Fraction(1)) -> None:
    for word, coefficient in source:
        target[word] += scale * coefficient


@lru_cache(maxsize=None)
def _reduce_word_cached(word: Word) -> tuple[tuple[Word, Fraction], ...]:
    inversion = next(
        (index for index in range(len(word) - 1) if _token_key(word[index]) > _token_key(word[index + 1])),
        None,
    )
    if inversion is None:
        return ((word, Fraction(1)),)
    index = inversion
    left, right = word[index], word[index + 1]
    out: defaultdict[Word, Fraction] = defaultdict(Fraction)
    swapped = word[:index] + (right, left) + word[index + 2 :]
    _combine(out, _reduce_word_cached(swapped))
    bracket_mode = left[3] + right[3]
    for base, coefficient in _bracket(_base(left), _base(right)):
        bracket_word = word[:index] + (_token(base, bracket_mode),) + word[index + 2 :]
        _combine(out, _reduce_word_cached(bracket_word), coefficient)
    return tuple(sorted(((key, value) for key, value in out.items() if value), key=lambda item: item[0]))


def _canonical_vector(
    terms: list[tuple[Fraction, Word]],
) -> dict[Word, Fraction]:
    out: defaultdict[Word, Fraction] = defaultdict(Fraction)
    for coefficient, word in terms:
        _combine(out, _reduce_word_cached(word), coefficient)
    return {word: coefficient for word, coefficient in out.items() if coefficient}


@lru_cache(maxsize=None)
def _act_one_cached(generator: Base, mode: int, word: Word) -> tuple[tuple[Word, Fraction], ...]:
    if not word:
        return ()
    first, rest = word[0], word[1:]
    out: defaultdict[Word, Fraction] = defaultdict(Fraction)

    # x(m) y(n)w = y(n)x(m)w + [x(m),y(n)]w
    for acted_word, coefficient in _act_one_cached(generator, mode, rest):
        _combine(out, _reduce_word_cached((first,) + acted_word), coefficient)

    commutator_mode = mode + first[3]
    for base, coefficient in _bracket(generator, _base(first)):
        if commutator_mode < 0:
            _combine(
                out,
                _reduce_word_cached((_token(base, commutator_mode),) + rest),
                coefficient,
            )
        else:
            _combine(out, _act_one_cached(base, commutator_mode, rest), coefficient)

    if commutator_mode == 0:
        central = Fraction(mode) * LEVEL * _bilinear(generator, _base(first))
        if central:
            out[rest] += central
    return tuple(sorted(((key, value) for key, value in out.items() if value), key=lambda item: item[0]))


def _act_vector(vector: dict[Word, Fraction], generator: Base, mode: int) -> dict[Word, Fraction]:
    out: defaultdict[Word, Fraction] = defaultdict(Fraction)
    for word, coefficient in vector.items():
        _combine(out, _act_one_cached(generator, mode, word), coefficient)
    return {word: coefficient for word, coefficient in out.items() if coefficient}


def _factor_dynkin(base: Base) -> tuple[int, int, int]:
    kind, i, j = base
    if kind == "h":
        return 0, 0, 0
    signs = [0, 0, 0, 0]
    signs[i - 1] += 1 if kind == "e" else -1
    signs[j - 1] -= 1 if kind == "e" else -1
    return tuple(signs[index] - signs[index + 1] for index in range(3))


def _degree_and_weights(vector: dict[Word, Fraction]) -> tuple[set[int], set[tuple[int, int, int]]]:
    degrees: set[int] = set()
    weights: set[tuple[int, int, int]] = set()
    for word in vector:
        degrees.add(sum(-factor[3] for factor in word))
        weight = [0, 0, 0]
        for factor in word:
            contribution = _factor_dynkin(_base(factor))
            weight = [left + right for left, right in zip(weight, contribution)]
        weights.add(tuple(weight))
    return degrees, weights


SINGULAR_GENERATORS: tuple[tuple[str, Base, int], ...] = (
    ("e12_0", ("e", 1, 2), 0),
    ("e23_0", ("e", 2, 3), 0),
    ("e34_0", ("e", 3, 4), 0),
    ("f14_1", ("f", 1, 4), 1),
)


# ---------------------------------------------------------------------------
# Polynomial spaces and affine lines
# ---------------------------------------------------------------------------


def _rref(rows: list[list[Fraction]]) -> tuple[tuple[Fraction, ...], ...]:
    matrix = [row[:] for row in rows if any(row)]
    if not matrix:
        return ()
    pivot_row = 0
    columns = len(matrix[0])
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, len(matrix)) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [entry / scale for entry in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row or not matrix[row][column]:
                continue
            scale = matrix[row][column]
            matrix[row] = [entry - scale * pivot_entry for entry, pivot_entry in zip(matrix[row], matrix[pivot_row])]
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return tuple(tuple(row) for row in matrix if any(row))


def _polynomial_space(
    polynomials: list[dict[tuple[int, int, int], Fraction]],
    monomials: list[tuple[int, int, int]],
) -> tuple[tuple[Fraction, ...], ...]:
    return _rref([[polynomial.get(monomial, Fraction()) for monomial in monomials] for polynomial in polynomials])


def _space_fraction(
    predicted: list[dict[tuple[int, int, int], Fraction]],
    reference: list[dict[tuple[int, int, int], Fraction]],
) -> tuple[float, dict[str, int]]:
    monomials = sorted(set().union(*(set(poly) for poly in predicted + reference)))
    pspace = _polynomial_space(predicted, monomials)
    rspace = _polynomial_space(reference, monomials)
    combined = _rref([list(row) for row in pspace + rspace])
    intersection = len(pspace) + len(rspace) - len(combined)
    extra = max(0, len(pspace) - intersection)
    fraction = max(0.0, (intersection - 0.5 * extra) / max(1, len(rspace)))
    return min(1.0, fraction), {
        "predicted_rank": len(pspace),
        "reference_rank": len(rspace),
        "intersection_dimension": intersection,
        "extra_dimension": extra,
    }


def _line_key(
    family: tuple[tuple[Fraction, Fraction, Fraction], tuple[Fraction, Fraction, Fraction]],
) -> tuple[tuple[Fraction, Fraction, Fraction], tuple[Fraction, Fraction, Fraction]]:
    offset, direction = family
    pivot = next(index for index, entry in enumerate(direction) if entry)
    normalized_direction = tuple(entry / direction[pivot] for entry in direction)
    normalized_offset = tuple(
        entry - offset[pivot] * direction_entry
        for entry, direction_entry in zip(offset, normalized_direction)
    )
    return normalized_offset, normalized_direction


def _line_fraction(predicted: list[Any], reference: list[Any]) -> tuple[float, dict[str, int]]:
    pred_keys = [_line_key(item) for item in predicted]
    ref_keys = [_line_key(item) for item in reference]
    pred_set, ref_set = set(pred_keys), set(ref_keys)
    matched = len(pred_set & ref_set)
    duplicate_penalty = len(pred_keys) - len(pred_set)
    extra = len(pred_set - ref_set) + duplicate_penalty
    fraction = max(0.0, (matched - 0.5 * extra) / max(1, len(ref_set)))
    return min(1.0, fraction), {
        "matched_lines": matched,
        "reference_lines": len(ref_set),
        "extra_or_duplicate_lines": extra,
    }


def _poly_multiply(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    out = [Fraction()] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            out[i + j] += a * b
    return out


def _line_satisfies_polynomial(
    family: tuple[tuple[Fraction, Fraction, Fraction], tuple[Fraction, Fraction, Fraction]],
    polynomial: dict[tuple[int, int, int], Fraction],
) -> bool:
    offset, direction = family
    result: list[Fraction] = [Fraction()]
    for exponents, coefficient in polynomial.items():
        term = [coefficient]
        for index, exponent in enumerate(exponents):
            factor = [offset[index], direction[index]]
            power = [Fraction(1)]
            for _ in range(exponent):
                power = _poly_multiply(power, factor)
            term = _poly_multiply(term, power)
        if len(result) < len(term):
            result.extend([Fraction()] * (len(term) - len(result)))
        for index, entry in enumerate(term):
            result[index] += entry
    return not any(result)


def _vector_fraction(predicted: dict[Word, Fraction], reference: dict[Word, Fraction]) -> tuple[float, dict[str, Any]]:
    common = sorted(set(predicted) & set(reference))
    if not common:
        return 0.0, {"matched_coordinates": 0, "reference_coordinates": len(reference), "extra_coordinates": len(predicted)}
    pivot = common[0]
    scale = reference[pivot] / predicted[pivot]
    scaled = {word: coefficient * scale for word, coefficient in predicted.items()}
    matched = sum(scaled.get(word) == coefficient for word, coefficient in reference.items())
    extra = sum(word not in reference or reference.get(word) != coefficient for word, coefficient in scaled.items())
    fraction = max(0.0, (matched - 0.5 * extra) / max(1, len(reference)))
    return min(1.0, fraction), {
        "matched_coordinates": matched,
        "reference_coordinates": len(reference),
        "extra_or_incorrect_coordinates": extra,
        "exact_up_to_scalar": scaled == reference,
    }


def _search_rows(payload: dict[str, Any]) -> dict[int, int]:
    return {
        row["conformal_weight"]: row["singular_nullity"]
        for row in payload["search"]["tested_weights"]
    }


def _pair(pred_dir: Path, ref_dir: Path, config: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    pred_file = str(config.get("result_file", "result.json"))
    ref_file = Path(str(config.get("reference_file", "result.json"))).name
    pred = _validate(_load(pred_dir / pred_file, reference=False), reference=False)
    ref = _validate(_load(ref_dir / ref_file, reference=True), reference=True)
    return pred, ref


def _score(name: str, fraction: float, weight: float, details: dict[str, Any]) -> ScoreDetail:
    fraction = max(0.0, min(1.0, float(fraction)))
    return ScoreDetail(
        scorer_name=name,
        score=weight * fraction,
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


@register_scorer("affine_sl4_output_contract")
class AffineSl4OutputContractScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred_file = str(config.get("result_file", "result.json"))
            payload = _validate(_load(pred_dir / pred_file, reference=False), reference=False)
            return _score(
                "affine_sl4_output_contract",
                1.0,
                weight,
                {
                    "vector_terms": len(payload["singular_vector"]["terms"]),
                    "polynomial_count": len(payload["zhu"]["polynomial_basis"]),
                    "family_count": len(payload["category_o"]["families"]),
                },
            )
        except Exception as exc:
            return _failure("affine_sl4_output_contract", weight, exc)


@register_scorer("affine_sl4_search")
class AffineSl4SearchScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred, ref = _pair(pred_dir, ref_dir, config)
            pred_rows, ref_rows = _search_rows(pred), _search_rows(ref)
            matched = sum(pred_rows.get(degree) == nullity for degree, nullity in ref_rows.items())
            extra = len(set(pred_rows) - set(ref_rows))
            trace_fraction = max(0.0, (matched - 0.5 * extra) / max(1, len(ref_rows)))
            first = float(
                pred["search"]["first_nonvacuum_conformal_weight"]
                == ref["search"]["first_nonvacuum_conformal_weight"]
            )
            vector_dim = float(
                pred["singular_vector"]["solution_space_dimension"]
                == ref["singular_vector"]["solution_space_dimension"]
            )
            fraction = 0.75 * trace_fraction + 0.15 * first + 0.10 * vector_dim
            return _score(
                "affine_sl4_search",
                fraction,
                weight,
                {
                    "matched_search_rows": matched,
                    "reference_search_rows": len(ref_rows),
                    "extra_search_rows": extra,
                    "first_weight": first,
                    "solution_space_dimension": vector_dim,
                },
            )
        except Exception as exc:
            return _failure("affine_sl4_search", weight, exc)


@register_scorer("affine_sl4_singular_vector")
class AffineSl4SingularVectorScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred, ref = _pair(pred_dir, ref_dir, config)
            pred_terms = _terms(pred["singular_vector"]["terms"], "predicted vector", SubmissionError)
            ref_terms = _terms(ref["singular_vector"]["terms"], "reference vector", EvaluatorError)
            pred_vector = _canonical_vector(pred_terms)
            ref_vector = _canonical_vector(ref_terms)
            if not pred_vector:
                raise SubmissionError("the submitted singular vector reduces to zero")
            vector_fraction, vector_details = _vector_fraction(pred_vector, ref_vector)

            degrees, weights = _degree_and_weights(pred_vector)
            reported_degree = pred["search"]["first_nonvacuum_conformal_weight"]
            reported_weight = tuple(pred["singular_vector"]["finite_highest_weight_dynkin"])
            homogeneous = float(degrees == {reported_degree} and weights == {reported_weight})
            metadata = 0.5 * homogeneous + 0.5 * float(
                pred["singular_vector"]["finite_highest_weight_dynkin"]
                == ref["singular_vector"]["finite_highest_weight_dynkin"]
            )

            action_details: dict[str, int] = {}
            action_passes = 0
            for name, generator, mode in SINGULAR_GENERATORS:
                residual = _act_vector(pred_vector, generator, mode)
                action_details[name] = len(residual)
                action_passes += int(not residual)
            singularity = action_passes / len(SINGULAR_GENERATORS)
            reported_residuals = pred["singular_vector"]["annihilation_residual_term_counts"]
            report_consistency = sum(
                reported_residuals[name] == count for name, count in action_details.items()
            ) / len(action_details)

            fraction = 0.68 * vector_fraction + 0.20 * singularity + 0.07 * metadata + 0.05 * report_consistency
            # A near transcription that is not actually singular must not pass
            # the singular-vector component merely because most coordinates
            # happen to agree with the reference.
            if singularity < 1.0:
                fraction = min(fraction, 0.45)
            return _score(
                "affine_sl4_singular_vector",
                fraction,
                weight,
                {
                    **vector_details,
                    "canonical_coordinate_count": len(pred_vector),
                    "computed_degrees": sorted(degrees),
                    "computed_dynkin_weights": [list(weight_value) for weight_value in sorted(weights)],
                    "annihilation_residual_terms": action_details,
                    "annihilation_fraction": singularity,
                    "reported_residual_consistency": report_consistency,
                },
            )
        except Exception as exc:
            return _failure("affine_sl4_singular_vector", weight, exc)


@register_scorer("affine_sl4_zhu")
class AffineSl4ZhuScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred, ref = _pair(pred_dir, ref_dir, config)
            pred_polys = [
                _polynomial(item, f"predicted polynomial {index}", SubmissionError)
                for index, item in enumerate(pred["zhu"]["polynomial_basis"])
            ]
            ref_polys = [
                _polynomial(item, f"reference polynomial {index}", EvaluatorError)
                for index, item in enumerate(ref["zhu"]["polynomial_basis"])
            ]
            space_fraction, space_details = _space_fraction(pred_polys, ref_polys)
            zero_weight_dimension = float(
                pred["zhu"]["zero_weight_space_dimension"]
                == ref["zhu"]["zero_weight_space_dimension"]
            )
            mcp_count = float(pred["mcp_check_count"] == ref["mcp_check_count"])
            fraction = 0.85 * space_fraction + 0.10 * zero_weight_dimension + 0.05 * mcp_count
            return _score(
                "affine_sl4_zhu",
                fraction,
                weight,
                {**space_details, "zero_weight_space_dimension": zero_weight_dimension, "mcp_check_count": mcp_count},
            )
        except Exception as exc:
            return _failure("affine_sl4_zhu", weight, exc)


@register_scorer("affine_sl4_category_o")
class AffineSl4CategoryOScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            pred, ref = _pair(pred_dir, ref_dir, config)
            pred_families = [
                _family(item, f"predicted family {index}", SubmissionError)
                for index, item in enumerate(pred["category_o"]["families"])
            ]
            ref_families = [
                _family(item, f"reference family {index}", EvaluatorError)
                for index, item in enumerate(ref["category_o"]["families"])
            ]
            line_fraction, line_details = _line_fraction(pred_families, ref_families)
            pred_polys = [
                _polynomial(item, f"predicted polynomial {index}", SubmissionError)
                for index, item in enumerate(pred["zhu"]["polynomial_basis"])
            ]
            checks = [
                _line_satisfies_polynomial(family, polynomial)
                for family in pred_families
                for polynomial in pred_polys
            ]
            substitution_fraction = sum(checks) / max(1, len(checks))
            distinct_fraction = len({_line_key(family) for family in pred_families}) / len(pred_families)
            fraction = 0.85 * line_fraction + 0.10 * substitution_fraction + 0.05 * distinct_fraction
            return _score(
                "affine_sl4_category_o",
                fraction,
                weight,
                {
                    **line_details,
                    "submitted_lines": len(pred_families),
                    "polynomial_substitution_fraction": substitution_fraction,
                    "distinct_line_fraction": distinct_fraction,
                },
            )
        except Exception as exc:
            return _failure("affine_sl4_category_o", weight, exc)
