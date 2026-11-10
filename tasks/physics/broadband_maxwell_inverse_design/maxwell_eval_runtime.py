"""Deterministic discrete-physics kernel used by GT generation and scoring."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.signal import convolve2d
from scipy.sparse.linalg import spsolve


def pml_stretch(n: int, npml: int, strength: float, power: int) -> np.ndarray:
    depth = np.zeros(n, dtype=np.float64)
    if npml:
        depth[:npml] = np.arange(npml, 0, -1, dtype=float) / npml
        depth[-npml:] = np.arange(1, npml + 1, dtype=float) / npml
    return 1.0 + 1j * strength * depth**power


def assemble_helmholtz(
    eps_r: np.ndarray,
    omega: float,
    dx: float,
    dy: float,
    sx: np.ndarray,
    sy: np.ndarray,
) -> sparse.csr_matrix:
    eps = np.asarray(eps_r, dtype=np.complex128)
    ny, nx = eps.shape
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for j in range(ny):
        for i in range(nx):
            p = j * nx + i
            diag = omega**2 * eps[j, i] * sx[i] * sy[j]
            for di in (-1, 1):
                ii = i + di
                sx_face = sx[i] if ii < 0 or ii >= nx else 0.5 * (sx[i] + sx[ii])
                coeff = sy[j] / sx_face / dx**2
                diag -= coeff
                if 0 <= ii < nx:
                    rows.append(p)
                    cols.append(j * nx + ii)
                    vals.append(coeff)
            for dj in (-1, 1):
                jj = j + dj
                sy_face = sy[j] if jj < 0 or jj >= ny else 0.5 * (sy[j] + sy[jj])
                coeff = sx[i] / sy_face / dy**2
                diag -= coeff
                if 0 <= jj < ny:
                    rows.append(p)
                    cols.append(jj * nx + i)
                    vals.append(coeff)
            rows.append(p)
            cols.append(p)
            vals.append(diag)
    return sparse.coo_matrix(
        (vals, (rows, cols)), shape=(ny * nx, ny * nx), dtype=np.complex128
    ).tocsr()


def solve_field(
    eps_r: np.ndarray,
    omega: float,
    dx: float,
    dy: float,
    npml: int,
    pml_strength: float,
    pml_power: int,
    source: np.ndarray,
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = eps_r.shape
    sx = pml_stretch(nx, npml, pml_strength, pml_power)
    sy = pml_stretch(ny, npml, pml_strength, pml_power)
    stretch = sy[:, None] * sx[None, :]
    matrix = assemble_helmholtz(eps_r, omega, dx, dy, sx, sy)
    rhs = (-1j * omega * source * stretch).ravel(order="C")
    field = spsolve(matrix, rhs).reshape((ny, nx), order="C")
    return field, matrix, rhs, sx, sy


def _conic_kernel(radius: float) -> np.ndarray:
    extent = int(np.ceil(radius))
    yy, xx = np.mgrid[-extent : extent + 1, -extent : extent + 1]
    return np.maximum(0.0, radius - np.sqrt(xx * xx + yy * yy))


def density_filter(
    rho: np.ndarray, radius: float, beta: float, eta: float
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    kernel = _conic_kernel(radius)
    normalizer = convolve2d(
        np.ones_like(rho), kernel, mode="same", boundary="fill", fillvalue=0.0
    )
    filtered = convolve2d(
        rho, kernel, mode="same", boundary="fill", fillvalue=0.0
    ) / normalizer
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    projected = (
        np.tanh(beta * eta) + np.tanh(beta * (filtered - eta))
    ) / denominator
    return projected, {
        "kernel": kernel,
        "normalizer": normalizer,
        "filtered": filtered,
        "beta": float(beta),
        "eta": float(eta),
        "denominator": float(denominator),
    }


def density_filter_backward(
    upstream: np.ndarray, cache: dict[str, np.ndarray | float]
) -> np.ndarray:
    filtered = np.asarray(cache["filtered"])
    beta = float(cache["beta"])
    eta = float(cache["eta"])
    denominator = float(cache["denominator"])
    derivative = beta * (
        1.0 - np.tanh(beta * (filtered - eta)) ** 2
    ) / denominator
    weighted = upstream * derivative / np.asarray(cache["normalizer"])
    return convolve2d(
        weighted,
        np.asarray(cache["kernel"]),
        mode="same",
        boundary="fill",
        fillvalue=0.0,
    )


def material_from_rho(
    problem: dict, rho: np.ndarray, eta: float
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    projected, cache = density_filter(
        rho, float(problem["filter_radius"]), float(problem["beta"]), float(eta)
    )
    eps = np.asarray(problem["eps_background"], dtype=np.complex128).copy()
    mask = np.asarray(problem["design_mask"], dtype=bool)
    eps[mask] = float(problem["eps_min"]) + (
        float(problem["eps_max"]) - float(problem["eps_min"])
    ) * projected[mask]
    return eps, cache


def _case_gradient(
    problem: dict,
    rho: np.ndarray,
    field: np.ndarray,
    matrix: sparse.csr_matrix,
    sx: np.ndarray,
    sy: np.ndarray,
    omega: float,
    residual: np.ndarray,
    weight: np.ndarray,
    filter_cache: dict[str, np.ndarray | float],
) -> np.ndarray:
    adjoint_rhs = (weight * residual).ravel(order="C")
    adjoint = spsolve(matrix.conjugate().transpose().tocsr(), adjoint_rhs)
    stretch = sy[:, None] * sx[None, :]
    mass_field = omega**2 * stretch.ravel(order="C") * field.ravel(order="C")
    grad_eps = -np.real(np.conjugate(adjoint) * mass_field).reshape(
        rho.shape, order="C"
    )
    mask = np.asarray(problem["design_mask"], dtype=bool)
    upstream = np.zeros_like(rho, dtype=np.float64)
    upstream[mask] = grad_eps[mask] * (
        float(problem["eps_max"]) - float(problem["eps_min"])
    )
    grad = density_filter_backward(upstream, filter_cache)
    grad[~mask] = 0.0
    return grad


def evaluate_design(
    problem: dict, rho: np.ndarray, *, need_gradient: bool = True
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources = np.asarray(problem["sources"], dtype=np.complex128)
    targets = np.asarray(problem["targets"], dtype=np.complex128)
    weights = np.asarray(problem["weights"], dtype=np.float64)
    omegas = np.asarray(problem["omegas"], dtype=np.float64)
    etas = np.asarray(problem["projection_etas"], dtype=np.float64)
    nb, nf = len(etas), len(omegas)
    fields = np.empty((nb, nf, *rho.shape), dtype=np.complex128)
    case_losses = np.empty((nb, nf), dtype=np.float64)
    case_gradients: list[np.ndarray] = []
    last_eps = np.empty_like(np.asarray(problem["eps_background"]), dtype=np.complex128)

    for b, eta in enumerate(etas):
        eps, filter_cache = material_from_rho(problem, rho, float(eta))
        last_eps = eps
        for k, omega in enumerate(omegas):
            field, matrix, _, sx, sy = solve_field(
                eps,
                float(omega),
                float(problem["dx"]),
                float(problem["dy"]),
                int(problem["npml"]),
                float(problem["pml_strength"]),
                int(problem["pml_power"]),
                sources[k],
            )
            fields[b, k] = field
            residual = field - targets[b, k]
            weight = weights[b, k]
            case_losses[b, k] = 0.5 * float(np.sum(weight * np.abs(residual) ** 2))
            if need_gradient:
                case_gradients.append(
                    _case_gradient(
                        problem,
                        rho,
                        field,
                        matrix,
                        sx,
                        sy,
                        float(omega),
                        residual,
                        weight,
                        filter_cache,
                    )
                )

    q = float(problem["robust_order"])
    delta = float(problem["robust_loss_offset"])
    shifted = case_losses.ravel() + delta
    scale = float(np.mean(shifted**q) ** (1.0 / q))
    objective = scale - delta

    gradient = np.zeros_like(rho, dtype=np.float64)
    if need_gradient:
        coefficients = shifted ** (q - 1.0) / (shifted.size * scale ** (q - 1.0))
        for coeff, case_gradient in zip(coefficients, case_gradients, strict=True):
            gradient += float(coeff) * case_gradient
        gradient[~np.asarray(problem["design_mask"], dtype=bool)] = 0.0
    return float(objective), gradient, fields, case_losses, last_eps


def hessian_vector_product(problem: dict, rho: np.ndarray) -> np.ndarray:
    direction = np.asarray(problem["hvp_direction"], dtype=np.float64)
    step = float(problem["hvp_step"])
    plus = evaluate_design(problem, rho + step * direction, need_gradient=True)[1]
    minus = evaluate_design(problem, rho - step * direction, need_gradient=True)[1]
    hvp = (plus - minus) / (2.0 * step)
    hvp[~np.asarray(problem["design_mask"], dtype=bool)] = 0.0
    return hvp


def field_residuals(problem: dict, rho: np.ndarray, fields: np.ndarray) -> np.ndarray:
    etas = np.asarray(problem["projection_etas"], dtype=np.float64)
    omegas = np.asarray(problem["omegas"], dtype=np.float64)
    residuals = np.empty((len(etas), len(omegas)), dtype=np.float64)
    for b, eta in enumerate(etas):
        eps, _ = material_from_rho(problem, rho, float(eta))
        for k, omega in enumerate(omegas):
            _, matrix, rhs, _, _ = solve_field(
                eps,
                float(omega),
                float(problem["dx"]),
                float(problem["dy"]),
                int(problem["npml"]),
                float(problem["pml_strength"]),
                int(problem["pml_power"]),
                np.asarray(problem["sources"])[k],
            )
            vector = np.asarray(fields)[b, k].ravel(order="C")
            residuals[b, k] = np.linalg.norm(matrix @ vector - rhs) / max(
                float(np.linalg.norm(rhs)), 1e-15
            )
    return residuals
